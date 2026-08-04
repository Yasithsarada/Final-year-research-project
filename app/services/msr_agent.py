import asyncio
import logging
import httpx
import datetime
from typing import Dict, List, Any, Optional, Tuple

from app.core.config import settings
from app.models.github_metrics import (
    GitHubProfileMetrics,
    CommitHistory,
    PullRequestMetrics,
    RepositoryMetadata,
)

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"


class MSRAgent:
    """Mining Software Repositories (MSR) Evidence Agent.

    Fetches raw signals from the GitHub API:
      - Commit history  (frequency, size, message quality)
      - Pull Requests    (accepted / rejected ratios, comments)
      - Repository meta  (stars, forks, languages, followers)

    All requests are authenticated with GITHUB_TOKEN to get the higher
    rate-limit tier (5,000 req/hr instead of 60).
    """

    # GitHub computes /stats/* endpoints asynchronously and answers 202 with an
    # empty body until the cache is warm. Bounded retry so a cold repository
    # does not masquerade as an inactive one. 3 retries x 2s (6s total) was
    # enough for already-popular, frequently-queried repos (e.g. dabeaz's
    # OSS projects) but was not enough for smaller, rarely-queried personal
    # repositories — validated against 8 real accounts where several repos
    # never finished computing within the old budget, undercounting real
    # commit history.
    STATS_MAX_RETRIES = 6
    STATS_RETRY_DELAY_SECONDS = 3.0

    def __init__(self, stats_max_retries: Optional[int] = None, stats_retry_delay_seconds: Optional[float] = None):
        # Instance overrides fall back to the class defaults above, so the
        # production FastAPI app (which just calls MSRAgent()) is unaffected.
        # The ML pipeline passes app.ml.pipeline_config's STATS_MAX_RETRIES /
        # STATS_RETRY_DELAY_SECONDS explicitly instead of editing this file.
        self.stats_max_retries = stats_max_retries or self.STATS_MAX_RETRIES
        self.stats_retry_delay_seconds = stats_retry_delay_seconds or self.STATS_RETRY_DELAY_SECONDS

        headers = {"Accept": "application/vnd.github+json"}
        if settings.GITHUB_TOKEN:
            headers["Authorization"] = f"Bearer {settings.GITHUB_TOKEN}"
        # Shared async client — reused across all calls in one pipeline run.
        # follow_redirects is required: repositories that have been renamed or
        # transferred to an organisation (very common for popular OSS, e.g.
        # tiangolo/fastapi -> fastapi/fastapi) answer with 301. raise_for_status
        # does not treat 3xx as an error, so without this the redirect body was
        # parsed as if it were the stats payload, silently yielding zero commits.
        self._client = httpx.AsyncClient(
            base_url=GITHUB_API_BASE,
            headers=headers,
            timeout=20.0,
            follow_redirects=True,
        )

    async def _get(self, path: str, params: Optional[Dict] = None) -> Any:
        """Thin helper that raises on HTTP errors and returns parsed JSON.

        Handles GitHub's asynchronous statistics endpoints: the first request
        for a repository's contributor stats returns `202 Accepted` with an
        empty body while GitHub computes the cache. We retry a bounded number
        of times rather than treating it as "no activity", which would zero out
        the burst detector for any repo not already cached.
        """
        for attempt in range(self.stats_max_retries):
            resp = await self._client.get(path, params=params)
            resp.raise_for_status()

            if resp.status_code == 202 or not resp.content:
                logger.info(
                    f"GitHub is computing statistics for '{path}' "
                    f"(attempt {attempt + 1}/{self.stats_max_retries}); retrying."
                )
                await asyncio.sleep(self.stats_retry_delay_seconds)
                continue

            return resp.json()

        logger.warning(
            f"GitHub did not return statistics for '{path}' after "
            f"{self.stats_max_retries} attempts; treating as unavailable."
        )
        return None

    # ------------------------------------------------------------------
    # Commit History
    # ------------------------------------------------------------------

    async def _fetch_commit_history(
        self, username: str, repos: List[Dict]
    ) -> CommitHistory:
        """Aggregates commit statistics across a user's public repositories."""
        total_commits = 0
        total_additions = 0
        total_deletions = 0
        total_merge_commits = 0
        messages: List[str] = []
        # Sum commit counts per calendar week (unix week-start timestamp)
        # across every repo, so bursts that are spread over several small
        # repos still show up as a single concentrated spike instead of
        # being diluted into unrelated per-repo lists.
        weekly_counts_by_ts: Dict[int, int] = {}

        for repo in repos[:10]:  # Cap at 10 repos to respect rate limits
            repo_name = repo["full_name"]
            try:
                # Get contributor stats (includes weekly commit activity)
                stats = await self._get(f"/repos/{repo_name}/stats/contributors")
                if isinstance(stats, list):
                    for contributor in stats:
                        if not contributor:
                            continue
                        # Same null-vs-missing-key trap as the commits loop
                        # below: GitHub returns "author": null for
                        # unresolvable/anonymous contributors, so .get(key,
                        # default) doesn't help — the key IS present.
                        author_login = (contributor.get("author") or {}).get("login", "")
                        if author_login.lower() == username.lower():
                            for week in contributor.get("weeks", []):
                                week_count = week.get("c", 0)
                                week_ts = week.get("w", 0)
                                total_commits += week_count
                                total_additions += week.get("a", 0)
                                total_deletions += week.get("d", 0)
                                if week_count > 0:
                                    weekly_counts_by_ts[week_ts] = (
                                        weekly_counts_by_ts.get(week_ts, 0) + week_count
                                    )

                # Sample recent commit messages
                commits = await self._get(
                    f"/repos/{repo_name}/commits",
                    params={"author": username, "per_page": 10},
                )
                for c in commits or []:
                    if not c:
                        continue
                    # dict.get(key, default) only falls back when the key is
                    # MISSING — GitHub sometimes returns "commit": null (e.g.
                    # ghost-author / redacted commits), where the key exists
                    # but its value is None, so .get("commit", {}) returns
                    # None itself and a chained .get("message") crashes.
                    msg = (c.get("commit") or {}).get("message", "")
                    if msg:
                        messages.append(msg[:120])
                    # Detect merge commits
                    parents = c.get("parents") or []
                    if len(parents) > 1:
                        total_merge_commits += 1

            except Exception as e:
                logger.warning(f"Could not fetch commits for {repo_name}: {e}")
                continue

        # Build the full weekly time series (including zero-activity weeks)
        # spanning from the first to the last active week, so the Fraud &
        # Anomaly Detection Model can see the true shape of the activity
        # curve rather than just an average.
        weekly_commit_counts: List[int] = []
        if weekly_counts_by_ts:
            sorted_ts = sorted(weekly_counts_by_ts.keys())
            WEEK_SECONDS = 7 * 24 * 60 * 60
            first_ts, last_ts = sorted_ts[0], sorted_ts[-1]
            n_weeks = max(int(round((last_ts - first_ts) / WEEK_SECONDS)) + 1, 1)
            weekly_commit_counts = [
                weekly_counts_by_ts.get(first_ts + i * WEEK_SECONDS, 0)
                for i in range(n_weeks)
            ]

        active_weeks = [c for c in weekly_commit_counts if c > 0]
        # "Active-week" average — used elsewhere as a coarse activity signal.
        freq = sum(active_weeks) / max(len(active_weeks), 1)
        avg_size = (
            (total_additions + total_deletions) / max(total_commits, 1)
        )
        merge_pct = (
            (total_merge_commits / max(total_commits, 1)) * 100
        )

        return CommitHistory(
            total_commits=total_commits,
            frequency_per_week=round(freq, 2),
            average_commit_size_lines=round(avg_size, 2),
            merging_percentage=round(merge_pct, 2),
            commit_messages=messages[:20],
            weekly_commit_counts=weekly_commit_counts,
        )

    # ------------------------------------------------------------------
    # Pull Requests
    # ------------------------------------------------------------------

    async def _fetch_pr_metrics(
        self, username: str, repos: List[Dict]
    ) -> PullRequestMetrics:
        """Aggregates pull-request quality signals."""
        total_prs = 0
        accepted = 0
        rejected = 0
        total_comments = 0

        for repo in repos[:10]:
            repo_name = repo["full_name"]
            try:
                # Fetch closed PRs authored by this user
                prs = await self._get(
                    f"/repos/{repo_name}/pulls",
                    params={"state": "closed", "per_page": 30},
                )
                for pr in prs or []:
                    if pr.get("user", {}).get("login", "").lower() != username.lower():
                        continue
                    total_prs += 1
                    total_comments += pr.get("comments", 0) + pr.get("review_comments", 0)
                    if pr.get("merged_at"):
                        accepted += 1
                    else:
                        rejected += 1
            except Exception as e:
                logger.warning(f"Could not fetch PRs for {repo_name}: {e}")
                continue

        approval_rate = round(accepted / max(total_prs, 1), 3)
        avg_comments = round(total_comments / max(total_prs, 1), 2)

        return PullRequestMetrics(
            total_prs=total_prs,
            accepted_prs=accepted,
            rejected_prs=rejected,
            approval_rate=approval_rate,
            average_comments_per_pr=avg_comments,
        )

    # ------------------------------------------------------------------
    # Repository Metadata
    # ------------------------------------------------------------------

    async def _fetch_repo_metadata(
        self, username: str, repos: List[Dict]
    ) -> RepositoryMetadata:
        """Aggregates repository-level statistics."""
        total_stars = sum(r.get("stargazers_count", 0) for r in repos)
        total_forks = sum(r.get("forks_count", 0) for r in repos)
        language_counts: Dict[str, int] = {}

        for repo in repos:
            lang = repo.get("language")
            if lang:
                language_counts[lang] = language_counts.get(lang, 0) + 1

        # Sort by frequency descending
        sorted_langs = dict(
            sorted(language_counts.items(), key=lambda x: x[1], reverse=True)
        )

        try:
            user_info = await self._get(f"/users/{username}") or {}
            followers = user_info.get("followers", 0)
            collaborations = user_info.get("public_repos", 0)
        except Exception:
            followers = 0
            collaborations = len(repos)

        return RepositoryMetadata(
            total_repos=len(repos),
            total_stars=total_stars,
            total_forks=total_forks,
            primary_languages=sorted_langs,
            followers=followers,
            collaborations=collaborations,
        )

    # ------------------------------------------------------------------
    # Public Entry Point
    # ------------------------------------------------------------------

    async def analyze_profile(self, github_url_or_username: str) -> GitHubProfileMetrics:
        """Full MSR analysis pipeline for a GitHub user.

        Args:
            github_url_or_username: Either a full GitHub URL (e.g.
                https://github.com/torvalds) or just a username.

        Returns:
            GitHubProfileMetrics with all aggregated signals.
        """
        # Parse username from URL if needed
        username = github_url_or_username.rstrip("/").split("/")[-1]
        logger.info(f"MSR Agent: Starting analysis for GitHub user '{username}'")

        try:
            # 1. Fetch all public repos
            repos = await self._get(
                f"/users/{username}/repos",
                params={"type": "public", "per_page": 100, "sort": "updated"},
            )
            if not isinstance(repos, list):
                repos = []
            logger.info(f"MSR Agent: Found {len(repos)} public repos for '{username}'")

            # 2. Fetch all signals in sequence (GitHub has strict concurrency limits)
            commit_history = await self._fetch_commit_history(username, repos)
            pr_metrics = await self._fetch_pr_metrics(username, repos)
            repo_metadata = await self._fetch_repo_metadata(username, repos)

            logger.info(f"MSR Agent: Analysis complete for '{username}'")
            return GitHubProfileMetrics(
                username=username,
                commit_history=commit_history,
                pull_requests=pr_metrics,
                repository_metadata=repo_metadata,
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"MSR Agent: GitHub API error for '{username}': {e.response.status_code} {e.response.text}")
            raise
        finally:
            await self._client.aclose()
