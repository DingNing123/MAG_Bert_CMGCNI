import os
import csv 

def convert_bytes(size, unit=None):

    if unit == "KB":
        return print('File size: ' + str(round(size / 1024, 3)) + ' Kilobytes')
    elif unit == "MB":
        return print('File size: ' + str(round(size / (1024 * 1024), 3)) + ' Megabytes')
    elif unit == "GB":
        return print('File size: ' + str(round(size / (1024 * 1024 * 1024), 3)) + ' Gigabytes')
    else:
        return print('File size: ' + str(size) + ' bytes')

def get_file_size(file_path,unit=None):
    size = os.path.getsize(file_path)
    convert_bytes(size,unit)


def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " %(key, src_dict[key]) 
    return dst_str

def csv_header(path, fieldnames):
    with open(path, mode='w', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file,fieldnames = fieldnames)
        writer.writeheader()


def csv_add_one_row(path, fieldnames, row):
    with open(path,  mode='a+', encoding='utf-8',newline='') as csv_file:
        writer = csv.DictWriter(csv_file,fieldnames = fieldnames)
        writer.writerow(row)

class Storage(dict):
    """
    A Storage object is like a dictionary except `obj.foo` can be used inadition to `obj['foo']`
    ref: https://blog.csdn.net/a200822146085/article/details/88430450
    """
    def __getattr__(self, key):
        try:
            return self[key] if key in self else False
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __str__(self):
        return "<" + self.__class__.__name__ + dict.__repr__(self) + ">"

