# -*- mode: python -*-

load("@drake//tools/workspace:github.bzl", "github_archive")

def ibex_repository(
        name,
        mirrors = None):
    github_archive(
        name = name,
        # TODO(russt): soonhokong recommends we consider moving this to
        # RobotLocomotion.
        repository = "dreal-deps/ibex-lib",
        # As discussed in #15872, we need ibex <= 2.8.7 for CLP support.
        commit = "ibex-2.8.6_4",
        sha256 = "172f2cf8ced69bd2e30be448170655878735af7d0bf6d2fef44b14215c8b1a49",  # noqa
        build_file = "@drake//tools/workspace/ibex:package.BUILD.bazel",
        mirrors = mirrors,
    )
