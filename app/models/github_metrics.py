from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class CommitHistory(BaseModel):
    total_commits: int = 0
    frequency_per_week: float = 0.0
    average_commit_size_lines: float = 0.0
    merging_percentage: float = 0.0
    commit_messages: List[str] = Field(default_factory=list)
    # Full per-week commit counts (including zero-activity weeks), aggregated
    # across all analysed repos. This is the raw time series that the Fraud &
    # Anomaly Detection Model uses to tell genuine, sustained activity apart
    # from a short burst of commits made right before applying for a job.
    weekly_commit_counts: List[int] = Field(default_factory=list)

class PullRequestMetrics(BaseModel):
    total_prs: int = 0
    accepted_prs: int = 0
    rejected_prs: int = 0
    approval_rate: float = 0.0
    average_comments_per_pr: float = 0.0

class RepositoryMetadata(BaseModel):
    total_repos: int = 0
    total_stars: int = 0
    total_forks: int = 0
    primary_languages: Dict[str, int] = Field(default_factory=dict)
    followers: int = 0
    collaborations: int = 0

class GitHubProfileMetrics(BaseModel):
    username: str
    commit_history: CommitHistory = Field(default_factory=CommitHistory)
    pull_requests: PullRequestMetrics = Field(default_factory=PullRequestMetrics)
    repository_metadata: RepositoryMetadata = Field(default_factory=RepositoryMetadata)
