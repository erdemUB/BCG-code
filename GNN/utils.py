import os


def get_split_info(args):
    # if args['malnet_tiny']:
    #     args['group'] = 'type'

    # split_dir = 'split-info-tiny' if args['malnet_tiny'] else 'split-info'
    # data_dir = args['malnet_tiny_dir'] if args['malnet_tiny'] else args['malnet_dir']

    data_dir = args['data_dir']
    root_dir = '/projects/academic/erdem/jakir/easybuild/malware_classification/code/malnet-graph/split_info/'
    split_dir = root_dir + args['data_type'] + "/" + args['group']
    if args['rem_dup']:
        split_dir = root_dir + args['data_type'] + "/" + args['group'] + '_unique'

    print("split_dir ===   ", split_dir)
    # with open(os.getcwd() + '/../{}/{}/{}/train.txt'.format(split_dir, args['group'], args['train_ratio']), 'r') as f:
    #     lines_train = f.readlines()
    #
    # with open(os.getcwd() + '/../{}/{}/{}/val.txt'.format(split_dir, args['group'], args['train_ratio']), 'r') as f:
    #     lines_val = f.readlines()
    #
    # with open(os.getcwd() + '/../{}/{}/{}/test.txt'.format(split_dir, args['group'], args['train_ratio']), 'r') as f:
    #     lines_test = f.readlines()

    with open(split_dir + "/train.txt", 'r') as f:
        lines_train = f.readlines()

    with open(split_dir + "/val.txt", 'r') as f:
        lines_val = f.readlines()

    with open(split_dir + "/test.txt", 'r') as f:
        lines_test = f.readlines()


    print("data dir ==  ", data_dir, split_dir, args['group'])
    print(lines_val[:2])

    files_train = [data_dir + file.strip() + '.edgelist' for file in lines_train]
    files_val = [data_dir + file.strip() + '.edgelist' for file in lines_val]
    files_test = [data_dir + file.strip() + '.edgelist' for file in lines_test]

    # print(files_train[:2])

    print("Group name   ", args['group'])

    if args['group'] == 'type':
        print("inside type")
        graph_labels = sorted(list(set([file.split(data_dir)[1].split('/')[0] for file in files_train])))
        # type_labels = sorted(list(set([file.split(data_dir)[1].split('/')[1] for file in files_train])))
        # print("Label information   ", files_train[0], graph_labels[0], type_labels[0])

        label_dict = {t: idx for idx, t in enumerate(graph_labels)}
        print("label dictionary:  ", len(label_dict), label_dict)

        train_labels = [label_dict[file.split(data_dir)[1].split('/')[0]] for file in files_train]
        val_labels = [label_dict[file.split(data_dir)[1].split('/')[0]] for file in files_val]
        test_labels = [label_dict[file.split(data_dir)[1].split('/')[0]] for file in files_test]

        # print("Label len ===  ", len(train_labels), len(val_labels), len(test_labels))

    elif args['group'] == 'family':
        tmp_list = files_train.copy()
        tmp_list.extend(files_test)
        tmp_list.extend(files_val)
        graph_labels = sorted(list(set([file.split(data_dir)[1].split('/')[1] for file in tmp_list])))
        label_dict = {t: idx for idx, t in enumerate(graph_labels)}

        label_dict = {t: idx for idx, t in enumerate(graph_labels)}
        print("label dictionary:  ", len(label_dict), label_dict)

        train_labels = [label_dict[file.split(data_dir)[1].split('/')[1]] for file in files_train]
        val_labels = [label_dict[file.split(data_dir)[1].split('/')[1]] for file in files_val]
        test_labels = [label_dict[file.split(data_dir)[1].split('/')[1]] for file in files_test]

        # print(len(train_labels), train_labels)

    elif args['group'] == 'binary':
        graph_labels = ['benign', 'malicious']
        label_dict = {t: idx for idx, t in enumerate(graph_labels)}

        train_labels = [0 if 'benign' in file.split(data_dir)[1].split('/')[0] else 1 for file in files_train]
        val_labels = [0 if 'benign' in file.split(data_dir)[1].split('/')[0] else 1 for file in files_val]
        test_labels = [0 if 'benign' in file.split(data_dir)[1].split('/')[0] else 1 for file in files_test]

    else:
        print('Group does not exist')
        exit(1)

    print('Number of train samples: {}, val samples: {}, test samples: {}'.format(len(files_train), len(files_val), len(files_test)))

    return files_train, files_val, files_test, train_labels, val_labels, test_labels, label_dict