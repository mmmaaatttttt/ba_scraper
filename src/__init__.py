def clean_episode(filepath):
    chris_lines = []
    current_chris_section = []
    with open(filepath) as file:
        for line in file.readlines():
            if line.startswith("Chris:   "):
                current_chris_section.append(line)
            else:
                if len(current_chris_section) > 1:
                    chris_lines.append(current_chris_section)
                current_chris_section = []
    from IPython import embed; embed()


if __name__ == '__main__':
    clean_episode("./episodes/ba_01.txt")