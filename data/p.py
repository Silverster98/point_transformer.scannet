from natsort import natsorted
scene_train_list = []
scene_val_list = []
scene_test_list = []

with open('./scannetv2_train.txt', 'r') as f:
    for scene_id in f.readlines():
        scene_num = int(scene_id.strip()[5:9])
        view_num = int(scene_id.strip()[-2:])

        if view_num != 0:
            continue

        if scene_num >= 600:
            scene_test_list.append(scene_id)
        else:
            scene_train_list.append(scene_id)

with open('./scannetv2_val.txt', 'r') as f:
    for scene_id in f.readlines():
        scene_num = int(scene_id.strip()[5:9])
        view_num = int(scene_id.strip()[-2:])

        if view_num != 0:
            continue

        if scene_num >= 600:
            scene_test_list.append(scene_id)
        else:
            scene_val_list.append(scene_id)

with open('./scannetv2_train_new.txt', 'w') as fp:
    for l in scene_train_list:
        fp.write(l)

with open('./scannetv2_val_new.txt', 'w') as fp:
    for l in scene_val_list:
        fp.write(l)

with open('./scannetv2_test_new.txt', 'w') as fp:
    for l in scene_test_list:
        fp.write(l)


print(len(scene_train_list))
print(len(scene_val_list))
print(len(scene_test_list))

# print(natsorted(scene_train_list))
# print()
# print(natsorted(scene_val_list))
# print()
# print(natsorted(scene_test_list))