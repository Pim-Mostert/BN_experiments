from mldesigner import command_component, Input, Output

@command_component()
def hoi_component(
    name: Input(type="string"),
    output_file: Input(type="string"),
    output_dir: Output(type="uri_folder"),
):
    print(f'Hoi {name}')
    
    with open(f'{output_dir}/{output_file}', "w") as file:
        file.write(f'Groetjes van {name}')
 