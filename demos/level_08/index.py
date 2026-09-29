import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", app_title="Level 8 · Demo")


@app.cell
def _():
    from marimo_levels import app as _shared_lesson

    lesson_app = _shared_lesson.clone()
    return (lesson_app,)


@app.cell
def _():
    import marimo as _mo

    query_params = _mo.query_params()
    return (query_params,)


@app.cell(hide_code=True)
async def _(lesson_app, query_params):
    lesson = await lesson_app.embed(defs={"query_params": query_params, "lesson_number": 8})
    lesson.output
    return (lesson,)


if __name__ == "__main__":
    app.run()
