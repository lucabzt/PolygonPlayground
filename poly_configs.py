"""
Some styles for visualizing polygons with corner points
"""

class StyleDict(dict):
    """A dictionary that can be copied and updated easily."""

    def with_overrides(self, **overrides):
        """Return a new dict with overrides applied."""
        result = self.copy()
        result.update(overrides)
        return result

    def __call__(self, **overrides):
        """Shorthand for with_overrides"""
        return self.with_overrides(**overrides)


# Polygon styles as individual objects
DEFAULT = StyleDict(
    fill=False,
    edgecolor="blue",
    linewidth=3,
    zorder=5
)

SMALL = StyleDict(
    fill=False,
    edgecolor="blue",
    linewidth=1,
    zorder=2
)

WHITE = StyleDict(
    fill=True,
    fc=(0,0,0.7,0.2),
    edgecolor="white",
    linewidth=3,
    zorder=2
)

RED = StyleDict(
    fill=True,
    alpha=0.8,
    fc="red",
    edgecolor="yellow",
    linewidth=2,
    zorder=3
)

CUSTOM = StyleDict(
    fill=True,
    linewidth=2,
    zorder=3
)

# Corner marker styles as individual objects
CIRCLE_CORNERS = StyleDict(
    marker="o",
    s=80,  # Size
    color="white",
    linewidth=1,
    zorder=3,
    alpha=1.0
)

SMALL_CORNERS = StyleDict(
    marker="o",
    s=5,  # Size
    color="white",
    linewidth=1,
    zorder=3,
    alpha=1.0
)

SQUARE_CORNERS = StyleDict(
    marker="s",
    s=70,
    color="yellow",
    linewidth=1.5,
    zorder=4,
    alpha=0.9
)

DIAMOND_CORNERS = StyleDict(
    marker="D",
    s=100,
    color="lime",
    linewidth=1.5,
    zorder=5,
    alpha=1.0
)

NONE = StyleDict(visible=False)

DEFAULT_POLYS = [DEFAULT(edgecolor='lightgreen'), DEFAULT(edgecolor='darkblue'), SMALL(edgecolor='red', zorder=10)]
DEFAULT_EDGES = [SMALL_CORNERS(s=40, zorder=11), SMALL_CORNERS(s=40, zorder=11), NONE()]

