import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", app_title="Agent Kung-fu · Demo catalog")


@app.cell
def _():
    from marimo_app import app as _shared_catalog

    catalog_app = _shared_catalog.clone()
    return (catalog_app,)


@app.cell
def _():
    import marimo as _mo

    query_params = _mo.query_params()
    return (query_params,)


@app.cell(hide_code=True)
async def _(catalog_app, query_params):
    catalog = await catalog_app.embed(defs={"query_params": query_params})
    catalog.output
    return


if __name__ == "__main__":
    app.run()
