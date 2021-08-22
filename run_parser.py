from segd_parser import SegDParser
import os

if __name__ == '__main__':
    # segd_path = os.path.join('path', 'to', 'your', 'file.segd')
    segd_path = os.path.join(os.sep, 'ssd', 'gazprom', 'data', 'raw', '00000986.segd')

    # Usage example
    segd = SegDParser(segd_path)
    segd.read_segd()
    # Optional parameter output_dir, by default script creates new 'parsed' dir in segd directory
    segd.save_parsed_files()

    # print(segd.traces_data.shape)
    # print(len(segd.traceh_list))
