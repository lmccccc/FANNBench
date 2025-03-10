# 定义要修改的变量名和新值
import sys


if __name__ == "__main__":
    target_variable = sys.argv[1]
    new_value = sys.argv[2]
    vfile = "vars.sh"
    # 打开.sh文件进行读取
    with open(vfile, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    
    datasets = ["sift", "spacev", "redcaps", "youtube"]
    flags = ["# sift10m", "# spacev10m", "# redcaps1m", "# YT-RGB1m"]
    flag_map = {
        "sift": "# sift10m",
        "spacev": "# spacev10m",
        "redcaps": "# redcaps1m",
        "youtube": "# YT-RGB1m"
    }
    if target_variable == "dataset":
        assert new_value in datasets
        # 找到要修改的变量所在的位置
        ori_flag = flag_map[target_variable]
        new_flag = flag_map[new_value]
        print("modify dataset to: ", new_value)

    # 遍历每一行，查找要修改的变量
    for i, line in enumerate(lines):
        if "# all modify ends here, don't change the following code" in line:
            print("Error: not found target variable")
            break
        # 去除行尾换行符
        stripped_line = line.rstrip()
        # 找到等号的位置
        equal_index = stripped_line.find('=')
        if equal_index != -1:
            # 获取等号左边的部分
            var_part = stripped_line[:equal_index].strip()
            if var_part == target_variable:
                print(f'Found variable {target_variable} in line {i + 1}')
                print(f'Old value: {stripped_line[equal_index + 1:]}, New value: {new_value}')
                # 保留前置空格
                leading_spaces = line[:line.find(var_part)]
                # 构造新的行内容
                new_line = f'{leading_spaces}{target_variable}={new_value}\n'
                lines[i] = new_line
                break
        

        if target_variable in datasets:
            assert new_value in datasets
            # 找到要修改的变量所在的位置
            new_flag = flag_map[new_value]

            # add # to ori dataset
            for flag in flag_map.values():
                if flag in line and flag != new_flag:
                    idx = i+1
                    while len(lines[idx].strip()) > 0:
                        if lines[idx][0] != '#':
                            lines[idx] = "# " + lines[idx]
                        idx += 1

            # remove # for new dataset
            if new_flag in line:
                idx = i+1
                while len(lines[idx].strip()) > 0:
                    if lines[idx][0] == '#':
                        lines[idx] = lines[idx][1:].strip(" ")
                    idx += 1

        if "all modify ends here" in line:
            # end of adjustable variables
            break


    # 打开.sh文件进行写入，将修改后的内容写回
    with open(vfile, 'w', encoding='utf-8') as file:
        file.writelines(lines)

    print("modify done")