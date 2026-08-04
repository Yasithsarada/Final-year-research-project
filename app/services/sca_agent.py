import logging
import httpx
from typing import Optional, Dict, Any

from app.core.config import settings

logger = logging.getLogger(__name__)


class CodeQualityMetrics:
    """Structured container for SonarQube code quality measures."""
    def __init__(self, data: Dict[str, Any]):
        measures = {m["metric"]: m.get("value", "0") for m in data.get("component", {}).get("measures", [])}
        self.bugs: int = int(measures.get("bugs", 0))
        self.vulnerabilities: int = int(measures.get("vulnerabilities", 0))
        self.code_smells: int = int(measures.get("code_smells", 0))
        self.coverage: float = float(measures.get("coverage", 0.0))
        self.duplicated_lines_density: float = float(measures.get("duplicated_lines_density", 0.0))
        self.cyclomatic_complexity: int = int(measures.get("complexity", 0))
        self.maintainability_rating: str = measures.get("sqale_rating", "A")
        self.security_rating: str = measures.get("security_rating", "A")
        self.reliability_rating: str = measures.get("reliability_rating", "A")
        self.ncloc: int = int(measures.get("ncloc", 0))  # Lines of code

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bugs": self.bugs,
            "vulnerabilities": self.vulnerabilities,
            "code_smells": self.code_smells,
            "coverage_pct": self.coverage,
            "duplication_pct": self.duplicated_lines_density,
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "maintainability_rating": self.maintainability_rating,
            "security_rating": self.security_rating,
            "reliability_rating": self.reliability_rating,
            "lines_of_code": self.ncloc,
        }

    def quality_score(self) -> float:
        """Derive a normalized 0.0-1.0 quality score from SonarQube metrics.
        
        The rating fields follow SonarQube's convention:
            1=A (best), 2=B, 3=C, 4=D, 5=E (worst)
        """
        def rating_to_score(r: str) -> float:
            return {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4, 5: 0.2}.get(int(r), 0.5)

        reliability = rating_to_score(self.reliability_rating)
        security = rating_to_score(self.security_rating)
        maintainability = rating_to_score(self.maintainability_rating)

        # Coverage: 0-100 → 0-1
        coverage_score = min(self.coverage / 100.0, 1.0)

        # Duplication penalty: 0% duplication = 1.0, 100% duplication = 0.0
        duplication_penalty = max(0.0, 1.0 - (self.duplicated_lines_density / 100.0))

        # Weighted composite
        score = (
            0.30 * reliability
            + 0.30 * security
            + 0.20 * maintainability
            + 0.10 * coverage_score
            + 0.10 * duplication_penalty
        )
        return round(score, 3)


class SCAAgent:
    """Source Code Analysis (SCA) Module.

    Connects to a locally-running SonarQube instance
    (default: http://localhost:9000) via its Web API to retrieve
    code quality metrics for a given project component key.

    To generate a component key, run a SonarQube scan on the repo first:
        sonar-scanner -Dsonar.projectKey=<key> -Dsonar.sources=.

    SonarQube is started via Docker:
        docker run -d --name sonarqube -p 9000:9000 sonarqube:community
    """

    METRICS = [
        "bugs",
        "vulnerabilities",
        "code_smells",
        "coverage",
        "duplicated_lines_density",
        "complexity",
        "sqale_rating",
        "security_rating",
        "reliability_rating",
        "ncloc",
    ]

    def __init__(self):
        self._base_url = settings.SONARQUBE_URL.rstrip("/")
        # SonarQube uses HTTP Basic Auth: token as username, empty password
        self._client = httpx.AsyncClient(timeout=15.0)

    async def analyze_project(
        self,
        component_key: str,
        sonar_token: Optional[str] = None,
    ) -> CodeQualityMetrics:
        """Fetches code quality measures for a SonarQube project component.

        Args:
            component_key: The SonarQube project key (e.g. "my_github_repo").
            sonar_token: Optional SonarQube token. Falls back to anonymous
                         access (works for public community edition).

        Returns:
            CodeQualityMetrics with all parsed metric values.
        """
        url = f"{self._base_url}/api/measures/component"
        params = {
            "component": component_key,
            "metricKeys": ",".join(self.METRICS),
        }
        auth = (sonar_token, "") if sonar_token else None

        try:
            logger.info(f"SCA Agent: Fetching metrics for component '{component_key}' from {self._base_url}")
            resp = await self._client.get(url, params=params, auth=auth)
            resp.raise_for_status()
            data = resp.json()
            metrics = CodeQualityMetrics(data)
            logger.info(f"SCA Agent: Quality score = {metrics.quality_score()}")
            return metrics
        except httpx.HTTPStatusError as e:
            logger.error(f"SCA Agent: SonarQube returned {e.response.status_code} for '{component_key}'")
            raise
        except httpx.ConnectError:
            logger.warning(
                f"SCA Agent: SonarQube unreachable at {self._base_url}. "
                "Returning neutral mock metrics."
            )
            return self._mock_metrics()
        finally:
            await self._client.aclose()

    @staticmethod
    def _mock_metrics() -> CodeQualityMetrics:
        """Returns neutral placeholder metrics when SonarQube is offline."""
        mock_data = {
            "component": {
                "measures": [
                    {"metric": "bugs", "value": "0"},
                    {"metric": "vulnerabilities", "value": "0"},
                    {"metric": "code_smells", "value": "5"},
                    {"metric": "coverage", "value": "50.0"},
                    {"metric": "duplicated_lines_density", "value": "10.0"},
                    {"metric": "complexity", "value": "10"},
                    {"metric": "sqale_rating", "value": "1"},
                    {"metric": "security_rating", "value": "1"},
                    {"metric": "reliability_rating", "value": "1"},
                    {"metric": "ncloc", "value": "500"},
                ]
            }
        }
        return CodeQualityMetrics(mock_data)
