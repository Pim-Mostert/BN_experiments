import argparse


def main():
    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    print(f'Hoi {args.name}')

    with open(f'{args.output_dir}/{args.output_file}', "w") as file:
        file.write(f'Groetjes van {args.name}')


if __name__ == "__main__":
    main()