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


class HeaderWriter:
    def __init__(self, gendir, tabwidth=4):
        self.incdir = os.path.join(gendir, "inc")
        os.makedirs(self.incdir, exist_ok=True)
        self.srcdir = os.path.join(gendir, "src")
        os.makedirs(self.srcdir, exist_ok=True)
        self.tabwidth = tabwidth

    def header_guard_begin(self, filename):
        guard = filename.replace(".", "_")
        return """#ifndef __{GUARD}__
#define __{GUARD}__

""".format(
            GUARD=guard.upper()
        )

    def header_guard_end(self, filename):
        guard = filename.replace(".", "_")
        return "#endif  // __{GUARD}__\n".format(GUARD=guard.upper())

    @property
    def includes(self):
        return "#include <pmsis.h>\n\n"

    def define(self, name, expr):
        if isinstance(expr, str):
            expr = f'"{expr}"'
        elif isinstance(expr, bool):
            expr = int(expr)
        expr = f"({expr})"
        return f"#define {name.upper()} {expr}\n"

    def vector_size(self, data):
        if hasattr(data, "numel"):
            return data.numel()
        elif hasattr(data, "size"):
            return data.size
        else:
            return len(data)

    def vector_declaration(self, name, size, _type):
        retval = ""
        retval += self.define(f"{name}_size", size)
        retval += f"{_type} {name}[{name.upper()}_SIZE]"
        return retval

    def vector_initial_value(self, data, elements_per_row=10):
        indent = " " * self.tabwidth
        size = self.vector_size(data)

        if hasattr(data, "flatten"):
            data = data.flatten()

        retval = ""
        retval += " = {"
        for i, element in enumerate(data):
            if i % elements_per_row == 0:
                retval += "\n" + indent
            retval += "{value:#04x}".format(value=int(element))
            if i < size - 1:
                retval += ", "
        retval += "\n}"
        return retval

    def vector_end(self):
        return ";\n\n"

    def render_vector(self, name, size, _type, init=None, elements_per_row=10):
        retval = ""
        retval += self.vector_declaration(name, _type, size)
        if init is not None:
            retval += self.vector_initial_value(init, elements_per_row)
        retval += self.vector_end()
        return retval

    def check_declaration(self, name):
        return f"void check_{name}();\n\n"

    def check(self, name):
        return f"""void check_{name}() {{
        printf("Checking the {name} vector:\\n");

        int n_err = 0;
        for (int i = 0; i < {name.upper()}_SIZE; i++) {{
            if ({name}[i] != golden_{name}[i]) {{
                printf("ERROR: wrong value of {name} @ %d: %d vs. golden: %d\\n", i, {name}[i], golden_{name}[i]);
                n_err++;
            }}
        }}

        if (n_err == 0)
            printf("> Success! No errors found.\\n");
        else
            printf("> Failure! Found %d/%d errors.\\n", n_err, {name.upper()}_SIZE);
    }}

    """

    def generate_header(self, name, body):
        filename = name + ".h"
        filepath = os.path.join(self.incdir, filename)

        print(f"Generating header file -> {filepath}")

        filerender = (
            self.header_guard_begin(filename) + body + self.header_guard_end(filename)
        )

        with open(filepath, "w") as file:
            file.write(filerender)

    def generate_vector_header(self, name, size, _type, init=None, golden=None):
        render = ""
        render += self.includes
        render += self.render_vector(name, "extern " + _type, size)

        if golden is not None:
            render += self.render_vector("golden_" + name, "extern " + _type, size)
            render += self.check_declaration(name)

        self.generate_header(name, render)

    def generate_source(self, name, body):
        filename = name + ".c"
        filepath = os.path.join(self.srcdir, filename)

        print(f"Generating source file -> {filepath}")

        with open(filepath, "w") as file:
            file.write(body)

    def generate_vector_source(self, name, size, _type, init=None, golden=None):
        render = ""
        render += f'#include "{name}.h"\n\n'
        render += self.render_vector(name, "PI_L1 " + _type, size, init=init)

        if golden is not None:
            render += self.render_vector(
                "golden_" + name, "PI_L1 " + _type, size, init=golden
            )
            render += self.check(name)

        self.generate_source(name, render)

    def generate_vector_files(self, name, size, _type, init=None, golden=None):
        self.generate_vector_source(name, size, _type, init, golden)
        self.generate_vector_header(name, size, _type, init, golden)

    def render_dims(self, name, dims):
        retval = ""
        for dim_name, dim_value in zip(dims["names"], dims["shape"]):
            retval += self.define(f"{name}_{dim_name}", dim_value)
        return retval

    def render_grouped_defines(self, defines, prefix=None):
        retval = ""
        for name, value in defines.items():
            full_name = name if prefix is None else f"{prefix}_{name}"
            if isinstance(value, dict):
                retval += self.render_grouped_defines(value, prefix=full_name)
                retval += "\n"
            else:
                retval += self.define(full_name, value)
        return retval

    def generate_defines_header(self, name, defines):
        self.generate_header(name, body=self.render_grouped_defines(defines) + "\n")
