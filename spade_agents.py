import asyncio
import os

from spade.agent import Agent
from spade.behaviour import OneShotBehaviour

from app import generate_comment_draft, support_report_for_user


class MonitoringAgent(Agent):
    async def get_monitoring(self, user_id: int) -> dict:
        return support_report_for_user(user_id)["monitoring"]


class DiagnosticAgent(Agent):
    async def get_diagnostics(self, user_id: int) -> dict:
        report = support_report_for_user(user_id)
        return {
            "risk_level": report["risk_level"],
            "risk_reasons": report["risk_reasons"],
            "area_stats": report["area_stats"],
        }


class RecommendationAgent(Agent):
    async def get_recommendations(self, user_id: int) -> list[str]:
        return support_report_for_user(user_id)["recommendations"]


class CommentDraftAgent(Agent):
    async def get_comment(self, user_id: int, username: str) -> str:
        report = support_report_for_user(user_id)
        return generate_comment_draft(
            username,
            report["monitoring"],
            report["risk_level"],
            report["recommendations"],
        )


class DemoBehaviour(OneShotBehaviour):
    async def run(self) -> None:
        user_id = int(os.environ.get("SPADE_USER_ID", "1"))
        username = os.environ.get("SPADE_USERNAME", "student")
        monitoring = await self.agent.monitoring_agent.get_monitoring(user_id)
        diagnostics = await self.agent.diagnostic_agent.get_diagnostics(user_id)
        recommendations = await self.agent.recommendation_agent.get_recommendations(
            user_id
        )
        comment = await self.agent.comment_agent.get_comment(user_id, username)
        print("Мониторинг:", monitoring)
        print("Диагностика:", diagnostics)
        print("Рекомендации:", recommendations)
        print("Комментарий:", comment)
        await self.agent.stop()


class OrchestratorAgent(Agent):
    async def setup(self) -> None:
        self.monitoring_agent = MonitoringAgent(
            os.environ.get("SPADE_MONITOR_JID", "monitor@localhost"),
            os.environ.get("SPADE_MONITOR_PASSWORD", "monitor"),
        )
        self.diagnostic_agent = DiagnosticAgent(
            os.environ.get("SPADE_DIAG_JID", "diag@localhost"),
            os.environ.get("SPADE_DIAG_PASSWORD", "diag"),
        )
        self.recommendation_agent = RecommendationAgent(
            os.environ.get("SPADE_REC_JID", "rec@localhost"),
            os.environ.get("SPADE_REC_PASSWORD", "rec"),
        )
        self.comment_agent = CommentDraftAgent(
            os.environ.get("SPADE_COMMENT_JID", "comment@localhost"),
            os.environ.get("SPADE_COMMENT_PASSWORD", "comment"),
        )
        self.add_behaviour(DemoBehaviour())


async def main() -> None:
    orchestrator = OrchestratorAgent(
        os.environ.get("SPADE_ORCH_JID", "orch@localhost"),
        os.environ.get("SPADE_ORCH_PASSWORD", "orch"),
    )
    await orchestrator.start()
    await asyncio.sleep(2)
    await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(main())
