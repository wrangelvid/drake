# -*- mode: python -*-
# vi: set ft=python :

load(
    "@drake//tools/workspace:github.bzl",
    "github_archive",
)

def common_robotics_utilities_repository(
        name,
        mirrors = None):
    github_archive(
        name = name,
        repository = "wrangelvid/common_robotics_utilities",
        upgrade_advice = """
        When updating, ensure that any new unit tests are reflected in
        package.BUILD.bazel and BUILD.bazel in drake. Tests may have been
        updated in wrangelvid/common_robotics_utilities/test/ or
        wrangelvid/common_robotics_utilities/CMakeLists.txt.ros2
        """,
        commit = "a0bbfc0191d5bf065d7bd4edc9f80da9ce9c0900",
        sha256 = "a8170d94620bbdcb6525ee64aac9b87b260f430aa8e9cd4cf39628ccbf89cb5c",  # noqa
        build_file = ":package.BUILD.bazel",
        mirrors = mirrors,
    )
