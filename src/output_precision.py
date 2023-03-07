from string import Template
import argparse, os

import pandas as pd


def change_precision(submission: str, output_dir: str)->None:
    file = submission.split('/')[-1]
    file_split = file.split('.')
    output = Template(output_dir + file_split[-2] + '_$decimal.' + file_split[-1])

    df = pd.read_csv(submission)

    for i in range(3):
        df['Predicted Rice Yield (kg/ha)'] = df['Predicted Rice Yield (kg/ha)'].round(i)
        df.to_csv(output.substitute(decimal=i))


def get_args(default_args: dict):
    parser = argparse.ArgumentParser(description="Given a submission file, create 3 others submission file with the target rounded to the unit, the tenth and the hundredth.")

    parser.add_argument('--submission', nargs=1, type=str, metavar='str', default=default_args['submission'], help=f'Submission file. By default his param is equal to "{default_args["submission"]}"')
    parser.add_argument('--output_dir', nargs=1, type=str, metavar='str', default=default_args['output_dir'], help=f'Output directory for the new submissions files. By default his param is equal to "{default_args["output_dir"]}"')

    args = parser.parse_args()

    if not (os.path.exists(args.submission) and os.path.isfile(args.submission)):
        args.submission = default_args['submission']
    if not (os.path.exists(args.output_dir) and os.path.isdir(args.output_dir)):
        args.output_dir = default_args['output_dir']

    return args

if __name__ == "__main__":
    default_args = {
        "submission": "notebooks/model/adaptative/submission.csv",
        "output_dir": "notebooks/model/adaptative/"
    }
    args = get_args(default_args)
    change_precision(args.submission, args.output_dir)