
import torch


def get_random_problems(batch_size, node_cnt, problem_gen_params):

   ################################
    # "tmat" type
    ################################

    int_min = problem_gen_params['int_min']
    int_max = problem_gen_params['int_max']
    scaler = problem_gen_params['scaler']

    problems = torch.randint(low=int_min, high=int_max, size=(batch_size, node_cnt, node_cnt))
    # shape: (batch, node, node)
    problems[:, torch.arange(node_cnt), torch.arange(node_cnt)] = 0
    generate_tw = torch.randint(low= 1, high= 1000*1000, size=(batch_size, 2, node_cnt))
    while True:
        old_problems = problems.clone()

        problems, _ = (problems[:, :, None, :] + problems[:, None, :, :].transpose(2,3)).min(dim=3)
        # shape: (batch, node, node)

        if (problems == old_problems).all():
            break

    # Scale
    scaled_problems = problems.float() / scaler
    scaled_problems_time = scaled_problems.clone()
    start = generate_tw[:, 0, :]
    end = generate_tw[:, 1, :]
    return scaled_problems, scaled_problems_time,start,end
    # shape: (batch, node, node)


def load_single_problem_from_file(filename, node_cnt, scaler):

    ################################
    # "tmat" type
    ################################

    problem = torch.empty(size=(node_cnt, node_cnt), dtype=torch.long)
    # shape: (node, node)

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except Exception as err:
        print(str(err))

    line_cnt = 0
    for line in lines:
        linedata = line.split()

        if linedata[0].startswith(('TYPE', 'DIMENSION', 'EDGE_WEIGHT_TYPE', 'EDGE_WEIGHT_FORMAT', 'EDGE_WEIGHT_SECTION', 'EOF')):
            continue

        integer_map = map(int, linedata)
        integer_list = list(integer_map)

        problem[line_cnt] = torch.tensor(integer_list, dtype=torch.long)
        line_cnt += 1

    # Diagonals to 0
    problem[torch.arange(node_cnt), torch.arange(node_cnt)] = 0

    # Scale
    scaled_problem = problem.float() / scaler

    return scaled_problem
    # shape: (node, node)
