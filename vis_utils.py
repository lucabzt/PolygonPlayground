"""
This module is used to provide useful functions for visualization.
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from typing import Dict
from matplotlib.patches import Polygon
import numpy as np
from poly_configs import *

# Set dark background style
plt.style.use('dark_background')
plt.ion()


def get_plain(heading: str, size: int=500) -> plt.Figure:
    """
    :param heading: text written on the matplotlib plain
    :param size: size of the plain
    :return: matplotlib figure with a size x size plain to visualize polygons
    """
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(size / 100, size / 100), dpi=100)

    # Set the title of the plot
    ax.set_title(heading, fontsize=16, pad=16)

    # Add grid lines for better visualization
    ax.grid(True, linestyle='--', alpha=0.3, color='white')

    # Customize the plain
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)

    return fig


def draw_stats(ax, stats):
    ax.set_title("stats", color='white', pad=15, fontsize=40)
    ax.axis('off')  # Hide axes
    # Calculate text positions
    y_position = 0.8
    line_height = 0.06

    # Display the statistics
    for key, value in stats.items():
        # Format the display text
        if isinstance(value, float):
            display_text = f"{key}: {value:.2f}"
        else:
            display_text = f"{key}: {value}"

        # Add text
        ax.text(0.05, y_position, display_text,
                      fontsize=25, color='white',
                      transform=ax.transAxes)

        y_position -= line_height


def get_plain_with_stats(stats: Dict, h1: str = "Plot", h2: str = "Statistics", size: int = 500) -> plt.Figure:
    """
    Creates a plt figure with two axis, left are the polygons, right are the statistics like IOU
    :param stats: Dictionary containing statistics to display (keys as stat names, values as stat values)
    :param h1: Heading for the left plot (polygon visualization)
    :param h2: Heading for the right plot (statistics)
    :param size: Size of the plain for polygon visualization
    :return: Matplotlib figure with two subplots
    """
    # Create figure with custom grid layout - make the statistics panel more narrow
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

    # Left subplot for polygons (now proportionally larger)
    ax_plain = fig.add_subplot(gs[0])
    ax_plain.set_title(h1, color='white', pad=15, fontsize=40)
    ax_plain.set_xlim(0, size)
    ax_plain.set_ylim(0, size)
    ax_plain.set_aspect('equal')
    ax_plain.grid(True, linestyle='--', alpha=0.3, color='white')

    # Right subplot for statistics (now proportionally narrower)
    ax_stats = fig.add_subplot(gs[1])

    draw_stats(ax_stats, stats)

    # Add tight layout and spacing between subplots
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.2)  # Reduced wspace from 0.3 to 0.2 for less spacing

    return fig


def add_polygon(figure: plt.Figure, polygon, pol_style = DEFAULT(), cor_style = CIRCLE_CORNERS()) -> None:
    ax = figure.axes[0]

    # add the polygon
    mlp_polygon = Polygon(
        polygon,
        **pol_style
    )

    # add corners to the polygon
    if cor_style is not None:
        vertices_array = np.array(polygon)
        ax.scatter(
            vertices_array[:, 0],
            vertices_array[:, 1],
            **cor_style
        )

    figure.axes[0].add_patch(mlp_polygon)


def get_plot(polygons, stats, p_styles=None, c_styles=None) -> plt.Figure:
    plain = get_plain_with_stats(stats)
    for idx, poly in enumerate(polygons):
        p_style = DEFAULT() if p_styles is None else p_styles[idx]
        c_style = CIRCLE_CORNERS() if c_styles is None else c_styles[idx]
        if p_styles:
            p_style = p_styles[idx]
        if c_styles:
            c_style = c_styles[idx]

        add_polygon(plain, poly, p_style, c_style)

    return plain


def vis_poly(poly, p_style=SMALL(), c_style=SMALL_CORNERS()) -> None:
    plt.close()
    poly_plot = get_plain("Polygon")
    add_polygon(poly_plot, poly, p_style, c_style)
    plt.show()


def update_plot(fig, polys, stats, p_styles=None, c_styles=None):
    plot = fig.axes[0]
    st = fig.axes[1]
    patches_to_remove = [p for p in plot.patches]  # or lines, etc.
    for p in patches_to_remove:
        p.remove()
    # Remove any existing scatter points:
    for coll in plot.collections:
        coll.remove()
    for idx, poly in enumerate(polys):
        p_style = SMALL() if p_styles is None else p_styles[idx]
        c_style = SMALL_CORNERS() if c_styles is None else c_styles[idx]
        add_polygon(fig, poly, p_style, c_style)
    st.clear()
    draw_stats(st, stats)
    fig.canvas.draw()
    fig.canvas.flush_events()


if __name__ == '__main__':
    polygon1 = np.array([
        [120, 90],  # Top-left
        [280, 130],  # Top-right
        [310, 380],  # Bottom-right
        [70, 320]  # Bottom-left
    ])

    polygon2 = np.array([
        [350, 150],  # Top-left
        [450, 110],  # Top-right
        [420, 290],  # Bottom-right
        [310, 270]  # Bottom-left
    ])

    stats = {
        "IoU": 0.0,
        "DIoU": 0.05,
    }

    plt.close()
    f = get_plot([polygon1, polygon2], stats)
    plt.show()