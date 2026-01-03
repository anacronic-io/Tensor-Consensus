"""Curriculum learning scheduler for progressive difficulty."""

from typing import List, Dict, Any
import numpy as np


class CurriculumScheduler:
    """Curriculum scheduler for progressive training."""

    def __init__(
        self,
        stages: List[Dict[str, Any]],
        transition_threshold: float = 0.8,
    ):
        """Initialize curriculum scheduler.

        Args:
            stages: List of curriculum stages with configs
            transition_threshold: Performance threshold for stage transition
        """
        self.stages = stages
        self.transition_threshold = transition_threshold
        self.current_stage = 0

    def get_current_config(self) -> Dict[str, Any]:
        """Get configuration for current stage.

        Returns:
            Current stage configuration
        """
        return self.stages[self.current_stage]

    def should_advance(self, performance: float) -> bool:
        """Check if should advance to next stage.

        Args:
            performance: Current performance metric

        Returns:
            Whether to advance
        """
        return (
            performance >= self.transition_threshold and
            self.current_stage < len(self.stages) - 1
        )

    def advance(self):
        """Advance to next curriculum stage."""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1


def create_smac_curriculum() -> CurriculumScheduler:
    """Create curriculum for SMAC training.

    Returns:
        Curriculum scheduler
    """
    stages = [
        {'scenario': '3s_vs_5z', 'n_agents': 3, 'difficulty': '5'},
        {'scenario': '3s_vs_5z', 'n_agents': 3, 'difficulty': '7'},
        {'scenario': '27m_vs_30m', 'n_agents': 27, 'difficulty': '7'},
    ]

    return CurriculumScheduler(stages)
