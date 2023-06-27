import argparse as ap
import numpy as np


class ArgsDict:
    def __init__(self, args):
        parser = ap.ArgumentParser()
        for arg in args:
            parser.add_argument(
                "-{}".format(arg), nargs=1, help="{}".format(arg)
            )

        self.args_dict = vars(parser.parse_args())

    def ensure_arg(self, arg):
        if arg not in self.args_dict:
            raise Exception("No argument named {} was suplied.".format(arg))

    def get_string_list(self, arg):
        self.ensure_arg(arg)
        return [float(v) for v in self.args_dict[arg][0].split(" ")]

    def get_double_list(self, arg):
        return [float(v) for v in self.get_string_list(arg)]

    def get_double(self, arg):
        return self.get_double_list(arg)[0]

    def get_int(self, arg):
        return int(self.get_double(arg))


if __name__ == "__main__":
    print(ArgsDict(["NS", "T", "K"]).get_double_list("K"))
