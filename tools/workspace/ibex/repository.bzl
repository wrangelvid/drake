# -*- mode: python -*-

load("@drake//tools/workspace:github.bzl", "github_archive")

def ibex_repository(
        name,
        mirrors = None):
    github_archive(
        name = name,
        repository = "dreal-deps/ibex-lib",
        # As discussed in #15872, we need ibex <= 2.8.7 for CLP support.
        commit = "ibex-2.8.6_2",
        sha256 = "fef2b8592849bfbb69b83d9dad0afe8425a3f07da5515ed45c4ae4cf07859ee4",  # noqa
        build_file = "@drake//tools/workspace/ibex:package.BUILD.bazel",
        mirrors = mirrors,
    )
