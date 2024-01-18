# Luka Macan <luka.macan@unibo.it>
#
# Copyright 2023 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os
import re
from typing import Dict, Union, Optional, Tuple, Type
import locale
import subprocess
from Ne16 import Ne16
from Neureka import Neureka
from NnxTestClasses import NnxTest, NnxTestConf, NnxTestHeaderGenerator
from pathlib import Path

HORIZONTAL_LINE = "\n" + "-" * 100 + "\n"


def try_decode(stream: Optional[Union[bytes, str]]) -> Optional[str]:
    if stream is None:
        return None
    elif isinstance(stream, str):
        return stream
    elif isinstance(stream, bytes):
        _, encoding = locale.getlocale()
        assert encoding is not None, "ERROR: locale encoding is None"
        return stream.decode(encoding)
    else:
        assert False, f"ERROR: Unexpected datatype {type(stream)} of stream."


def captured_output(
    exception: Union[subprocess.CalledProcessError, subprocess.TimeoutExpired]
) -> Tuple[Optional[str], Optional[str]]:
    stdout = try_decode(exception.stdout)
    stderr = try_decode(exception.stderr)
    return stdout, stderr


def execute_command(
    cmd: str,
    timeout: int = 30,
    cflags: Optional[str] = None,
    envflags: Optional[Dict[str, str]] = None,
) -> Tuple[bool, str, str, Optional[str]]:
    env = os.environ
    if cflags:
        env["APP_CFLAGS"] = '"' + " ".join(cflags) + '"'
    if envflags:
        for key, value in envflags.items():
            env[key] = value

    status = None
    stdout = None

    try:
        proc = subprocess.run(
            cmd.split(),
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        status = True
        msg = "OK"
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.CalledProcessError as e:
        status = False
        msg = f"Build failed with exit status {e.returncode}."
        stdout, stderr = captured_output(e)
    except subprocess.TimeoutExpired as e:
        status = False
        msg = f"Timeout after {timeout}s."
        stdout, stderr = captured_output(e)

    if stdout is None:
        stdout = "<no stdout>"

    return status, msg, stdout, stderr


def assert_message(
    msg: str, test_name: str, cmd: str, stdout: str, stderr: Optional[str] = None
):
    retval = (
        f"Test {test_name} failed: {msg}\n"
        f"Command: {cmd}\n" + HORIZONTAL_LINE + f"\nCaptured stdout:\n{stdout}\n"
    )

    if stderr is not None:
        retval += f"\nCaptured stderr:\n{stderr}\n"

    return retval


def test(
    path: str,
    timeout: int,
    nnxName: str,
    nnxCls: Union[Type[Ne16], Type[Neureka]],
    nnxTestConfCls: Type[NnxTestConf],
):
    test_name = path
    test = NnxTest.load(nnxTestConfCls, path)

    NnxTestHeaderGenerator(nnxCls.weight_unroll).generate(test_name, test)

    Path("app/src/nnx_layer.c").touch()
    cmd = f"make -C app all run platform=gvsoc"
    passed, msg, stdout, stderr = execute_command(
        cmd=cmd, timeout=timeout, envflags={"ACCELERATOR": nnxName}
    )

    assert passed, assert_message(msg, test_name, cmd, stdout, stderr)

    match_success = re.search(r"> Success! No errors found.", stdout)
    match_fail = re.search(r"> Failure! Found (\d*)/(\d*) errors.", stdout)

    assert match_success or match_fail, assert_message(
        "No regexes matched.", test_name, cmd, stdout
    )

    assert not match_fail, assert_message(
        f"Errors found: {match_fail.group(1)}/{match_fail.group(2)}",
        test_name,
        cmd,
        stdout,
    )
