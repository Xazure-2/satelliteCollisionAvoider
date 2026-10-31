import reflex as rx

config = rx.Config(
    app_name="orbital",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.RadixThemesPlugin(theme=rx.theme(appearance="dark", accent_color="cyan", has_background=False)),
    ],
    telemetry_enabled=False,
    show_built_with_reflex=False,
)
