"""BA theme for Gradio (tested with gradio 6.31).

Usage:
    from theme import BATheme, CSS, HEADER_HTML
    with gr.Blocks() as demo:
        gr.HTML(HEADER_HTML)
        ...
    demo.launch(theme=BATheme(), css=CSS)      # gradio >= 6: theme/css go to launch()
    # gradio 4/5: gr.Blocks(theme=BATheme(), css=CSS)

The theme is always dark (sapphire ground). Light-mode and dark-mode variables are
set to the same values so the page looks the same whatever the visitor's OS setting.
See BA/docs/brand.md for the palette and the contrast ratios.
"""
from pathlib import Path

import gradio as gr
from gradio.themes.utils import colors, fonts, sizes

_HERE = Path(__file__).resolve().parent
CSS = (_HERE / "style.css").read_text(encoding="utf-8")
HEADER_HTML = (_HERE / "header.html").read_text(encoding="utf-8")

# Brand palette (BA/docs/brand.md)
SAPPHIRE = "#14254D"   # ground
DEEP = "#0F1D3E"       # input wells
SURFACE = "#1A2F5E"    # panels / blocks
LINE = "#2C477F"       # hairlines, dividers (decorative, never the only boundary)
RULE = "#6380B8"       # input and control borders (>= 3:1 on DEEP and SURFACE)
GOLD = "#D9B45A"       # primary action, the mark's line
GOLD_HI = "#E4C578"    # primary hover
TURQ = "#3FD0C9"       # focus, active, selection, links
IVORY = "#E8E3D3"      # body text
MUTED = "#A7B1C8"      # secondary text, labels
PLACEHOLDER = "#7F8BA8"  # placeholder text (inputs only, on DEEP)
ERROR = "#F08A7E"

gold = colors.Color(
    c50="#FBF6E9", c100="#F5EAC9", c200="#EEDBA3", c300="#E4C578", c400="#DEBC68",
    c500=GOLD, c600="#C29C42", c700="#9F7F33", c800="#7C6228", c900="#5A471D",
    c950="#3B2F13", name="ba_gold")
turquoise = colors.Color(
    c50="#E9FAF9", c100="#C9F3F0", c200="#9CE8E4", c300="#6EDDD7", c400="#55D6D0",
    c500=TURQ, c600="#2FB1AB", c700="#258C88", c800="#1C6B68", c900="#134B49",
    c950="#0B2E2D", name="ba_turquoise")
sapphire = colors.Color(
    c50="#F4F1E8", c100=IVORY, c200="#C9CEDC", c300=MUTED, c400=PLACEHOLDER,
    c500="#5A6A8E", c600=LINE, c700=SURFACE, c800=SAPPHIRE, c900=DEEP,
    c950="#0A142C", name="ba_sapphire")

FONT = [fonts.GoogleFont("IBM Plex Sans", weights=(400, 500, 600)),
        fonts.GoogleFont("IBM Plex Sans Arabic", weights=(400, 500, 600)),
        "ui-sans-serif", "system-ui", "Segoe UI", "Tahoma", "sans-serif"]
FONT_MONO = [fonts.GoogleFont("IBM Plex Mono", weights=(400,)),
             "ui-monospace", "SFMono-Regular", "Consolas", "monospace"]


class BATheme(gr.themes.Base):
    def __init__(self):
        super().__init__(
            primary_hue=gold, secondary_hue=turquoise, neutral_hue=sapphire,
            text_size=sizes.text_md, spacing_size=sizes.spacing_md, radius_size=sizes.radius_md,
            font=FONT, font_mono=FONT_MONO,
        )
        self.name = "ba"
        v = dict(
            # ground and text
            body_background_fill=SAPPHIRE,
            body_text_color=IVORY,
            body_text_color_subdued=MUTED,
            body_text_size="16px",
            background_fill_primary=SURFACE,
            background_fill_secondary=SAPPHIRE,
            border_color_primary=LINE,
            border_color_accent=TURQ,
            border_color_accent_subdued=LINE,
            color_accent=TURQ,
            color_accent_soft="rgba(63,208,201,0.14)",
            link_text_color=TURQ, link_text_color_hover="#6EDDD7",
            link_text_color_active=TURQ, link_text_color_visited=TURQ,
            code_background_fill=DEEP,
            shadow_drop="none", shadow_drop_lg="none", shadow_spread="3px",
            # blocks
            block_background_fill=SURFACE,
            block_border_color=LINE,
            block_border_width="1px",
            block_radius="14px",
            block_shadow="none",
            block_label_background_fill=SURFACE,
            block_label_border_color=LINE,
            block_label_text_color=MUTED,
            block_label_text_size="13px",
            block_label_text_weight="500",
            block_title_text_color=IVORY,
            block_title_text_weight="500",
            block_info_text_color=MUTED,
            panel_background_fill=SAPPHIRE,
            panel_border_color=LINE,
            layout_gap="18px",
            # inputs
            input_background_fill=DEEP,
            input_background_fill_focus=DEEP,
            input_background_fill_hover=DEEP,
            input_border_color=RULE,
            input_border_color_hover=MUTED,
            input_border_color_focus=TURQ,
            input_border_width="1px",
            input_placeholder_color=PLACEHOLDER,
            input_radius="10px",
            input_shadow="none",
            input_shadow_focus=f"0 0 0 2px rgba(63,208,201,0.35)",
            input_text_size="16px",
            # checkbox / radio
            checkbox_background_color=DEEP,
            checkbox_background_color_hover=DEEP,
            checkbox_background_color_focus=DEEP,
            checkbox_background_color_selected=TURQ,
            checkbox_border_color=RULE,
            checkbox_border_color_hover=MUTED,
            checkbox_border_color_focus=TURQ,
            checkbox_border_color_selected=TURQ,
            checkbox_border_width="1.5px",
            checkbox_label_background_fill=SURFACE,
            checkbox_label_background_fill_hover=SURFACE,
            checkbox_label_background_fill_selected=SURFACE,
            checkbox_label_border_color=LINE,
            checkbox_label_border_color_hover=RULE,
            checkbox_label_border_color_selected=TURQ,
            # sapphire tick/dot on turquoise (7.9:1) instead of Gradio's white one (1.7:1)
            checkbox_check=("url(\"data:image/svg+xml,%3csvg viewBox='0 0 16 16' fill='%2314254D' "
                            "xmlns='http://www.w3.org/2000/svg'%3e%3cpath d='M12.207 4.793a1 1 0 010 "
                            "1.414l-5 5a1 1 0 01-1.414 0l-2-2a1 1 0 011.414-1.414L6.5 9.086l4.293-4.293a1 "
                            "1 0 011.414 0z'/%3e%3c/svg%3e\")"),
            radio_circle=("url(\"data:image/svg+xml,%3csvg viewBox='0 0 16 16' fill='%2314254D' "
                          "xmlns='http://www.w3.org/2000/svg'%3e%3ccircle cx='8' cy='8' r='3'/%3e%3c/svg%3e\")"),
            checkbox_label_text_color=IVORY,
            checkbox_label_text_color_selected=IVORY,
            # buttons
            button_primary_background_fill=GOLD,
            button_primary_background_fill_hover=GOLD_HI,
            button_primary_border_color=GOLD,
            button_primary_border_color_hover=GOLD_HI,
            button_primary_text_color=SAPPHIRE,
            button_primary_text_color_hover=SAPPHIRE,
            button_primary_shadow="none", button_primary_shadow_hover="none",
            button_primary_shadow_active="none",
            button_secondary_background_fill="transparent",
            button_secondary_background_fill_hover="rgba(232,227,211,0.06)",
            button_secondary_border_color=RULE,
            button_secondary_border_color_hover=IVORY,
            button_secondary_text_color=IVORY,
            button_secondary_text_color_hover=IVORY,
            button_secondary_shadow="none", button_secondary_shadow_hover="none",
            button_cancel_background_fill="transparent",
            button_cancel_background_fill_hover="rgba(240,138,126,0.10)",
            button_cancel_border_color=ERROR,
            button_cancel_border_color_hover=ERROR,
            button_cancel_text_color=ERROR,
            button_cancel_text_color_hover=ERROR,
            button_border_width="1px",
            button_large_radius="10px", button_medium_radius="10px", button_small_radius="8px",
            button_large_text_weight="600", button_medium_text_weight="600",
            button_transform_hover="none", button_transform_active="none",
            # misc
            slider_color=TURQ,
            loader_color=TURQ,
            stat_background_fill=f"linear-gradient(to right, {TURQ}, {TURQ})",
            error_background_fill="rgba(240,138,126,0.10)",
            error_border_color=ERROR,
            error_text_color=ERROR,
            error_icon_color=ERROR,
            table_even_background_fill=SURFACE,
            table_odd_background_fill=SAPPHIRE,
            table_border_color=LINE,
        )
        # Same values in light and dark mode: BA is always on sapphire.
        both = {}
        for k, val in v.items():
            both[k] = val
            if hasattr(self, k + "_dark"):
                both[k + "_dark"] = val
        super().set(**both)
