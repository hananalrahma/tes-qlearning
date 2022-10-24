
import numpy as np

#mendefinisikan bentuk environment
env_rows = 11
env_columns = 11

q_values = np.zeros((env_rows, env_columns, 4))

#mendefinisikan action
actions = ['up', 'right', 'down', 'left']


#membangun environment
rewards = np.full((env_rows, env_columns), -100.)
rewards [0, 5] = 100.

aisles = {}
aisles[1] = [i for i in range (1, 10)]
aisles[2] = [1, 7, 9]
aisles[3] = [i for i in range(1, 9)]
aisles[3].append(9) 
aisles[4] = [3, 7]
aisles[5] = [i for i in range (11)]
aisles[6] = [5]
aisles[7] = [i for i in range(1, 10)]
aisles[8] = [3, 7]
aisles[9] = [i for i in range(11)]

#mengatur reward untuk tiap lorong
for row_index in range(1, 10):
    for column_index in aisles[row_index]:
        rewards[row_index, column_index] = -1

#menampilkan reward dalam bentuk matriks
for row in rewards:
    print(row)
##------------------------------------work-------------------------------------


def is_terminal_state(current_row_index, current_column_index):
    if rewards[current_row_index, current_column_index] == -1.:
        return False
    else:
        return True

def get_starting_location():
    current_row_index = np.random.randint(env_rows)
    current_column_index = np.random.randint(env_columns)
    while is_terminal_state(current_row_index, current_column_index):
        current_row_index = np.random.randint(env_rows)
        current_column_index = np.random.randint(env_columns)
    return current_row_index, current_column_index

def get_next_action(current_row_index, current_column_index, epsilon):
    if np.random.random() < epsilon:
        return np.argmax(q_values[current_row_index, current_column_index])
    else:
        return np.random.randint(4)

def get_next_location(current_row_index, current_column_index, action_index):
    new_row_index = current_row_index
    new_column_index = current_column_index
    if actions[action_index] == 'up' and current_row_index > 0:
        new_row_index -= 1
    elif actions[action_index] == 'right' and current_column_index < env_columns -1:
        new_column_index += 1
    elif actions[action_index] == 'down' and current_row_index < env_rows -1:
        new_row_index += 1
    elif actions[action_index] == 'left' and current_column_index > 0:
        new_column_index -= 1
    return new_row_index, new_column_index


def get_shortest_path(start_row_index, start_column_index):

    if is_terminal_state(start_row_index, start_column_index):
        return []
    else:
        current_row_index, current_column_index = start_row_index, start_column_index
        shortest_path = []
        shortest_path.append([current_row_index, current_column_index])

        while not is_terminal_state(current_row_index, current_column_index):
            action_index = get_next_action(current_row_index, current_column_index, 1.)

            current_row_index, current_column_index = get_next_location(current_row_index, current_column_index, action_index)
            shortest_path.append([current_row_index, current_column_index])
        return shortest_path

epsilon = 0.9
discount_factor = 0.9
learning_rate = 0.9

for episode in range (1000):
    row_index, column_index = get_starting_location()
    while not is_terminal_state(row_index, column_index):
        action_index = get_next_action(row_index, column_index, epsilon)

        old_row_index, old_column_index = row_index, column_index
        row_index, column_index = get_next_location(row_index, column_index, action_index)
        
        reward = rewards[row_index, column_index]
        old_q_values = q_values[old_row_index, old_column_index, action_index]
        temporal_difference = reward + (discount_factor * np.max(q_values[row_index, column_index])) - old_q_values

        new_q_value = old_q_values + (learning_rate * temporal_difference)
        q_values[old_row_index, old_column_index, action_index] = new_q_value

print('Training complete')

print(get_shortest_path(3, 9)) #dimulai pada baris ke 3, kolom 9
print("")
print(get_shortest_path(5, 0)) #dimulai dari baris ke 5, kolom 0
print("")
print(get_shortest_path(9, 5)) #dimulai dari baris ke 9, kolom 5

path = get_shortest_path(5, 2)
path.reverse()
print("")
print(path)


