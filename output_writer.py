

class OutputWriter:
    def __init__(self):
        pass

    def append_row(self, file_path, model_name, accuracy_data):
        out_str = model_name
        for acc in accuracy_data:
            out_str += "," + str(acc)
        out_str += "\n"
        with open(file_path, "a") as output:
            output.write(out_str)
