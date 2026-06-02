# Releasing statskeptic

This project publishes to PyPI from GitHub Actions using **Trusted Publishing**
(OIDC), so no API token or password is stored. The release workflow is
`.github/workflows/publish.yml`; it runs when a GitHub Release is published.

## One-time setup

1. Push the repository to GitHub as `Burton-David/statskeptic` (or update the
   `[project.urls]` in `pyproject.toml` and the values below to match the real path).
2. On GitHub: **Settings -> Environments -> New environment**, named `pypi`.
3. On PyPI: **Your account -> Publishing -> Add a new pending publisher**, with:

   | Field | Value |
   | --- | --- |
   | PyPI Project Name | `statskeptic` |
   | Owner | `Burton-David` |
   | Repository name | `statskeptic` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

   "Pending publisher" is the right choice for the first release, before the project
   exists on PyPI. After the first publish it becomes a normal Trusted Publisher.

## Cutting a release

1. Bump `version` in `pyproject.toml` (Semantic Versioning).
2. Add a section to `CHANGELOG.md`.
3. Commit, then tag and push:
   ```
   git commit -am "release: vX.Y.Z"
   git tag vX.Y.Z
   git push && git push --tags
   ```
4. On GitHub: **Releases -> Draft a new release**, choose the tag, publish it. The
   `publish` workflow builds the sdist + wheel and uploads them to PyPI.

## Dry run on TestPyPI (recommended for the first time)

Configure a second pending publisher on https://test.pypi.org with the same values,
then either publish a pre-release tag or run a one-off upload locally:

```
python -m build
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ statskeptic
```

## Manual fallback (API token)

If Trusted Publishing is not set up yet and you need to publish now, create a PyPI
API token (Account settings -> API tokens) and upload the built artifacts directly:

```
python -m build
twine check dist/*
twine upload dist/*        # username: __token__, password: the API token
```

## Before any publish

```
python -m build
twine check dist/*         # metadata and README render
```

A version can only be uploaded to PyPI once; bump the version for every release.
