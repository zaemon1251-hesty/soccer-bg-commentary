from tap import Tap


class Arguments(Tap):
    input_file: str
    output_file: str
    comment_csv_file: str
    query: str


def main(args: Arguments):
    pass

if __name__ == "__main__":
    ### construct query from the input file
    args = Arguments().parse_args()
    main(args)
