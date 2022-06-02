from CartPole import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse

def simulate(steps=1, initial_state=[0, 0, np.pi, 0], action=0, remap__angle=False):
    """
    Simulate the cartpole system for a number of specificed steps from a specified initial state.
    Returning an array containing all the states (at each step), including the initial state.
    """
    cp = CartPole() #Create CartPole object
    cp.setState(initial_state) #Initialise CartPole object with given initial state
    states = initial_state.copy() #Create copy of initial state array
    
    for step in range(steps): #PerformAction for a given number of steps
        cp.performAction(action)
        if remap__angle: #remap the angles to range [-pi, pi] if True
            cp.remap_angle()
        current_state = cp.getState() #Find state after one performAction
        states = np.vstack((states, current_state)) #Create stacked array with state after each performAction
    return states

def display_plots(states, model=False, model_states=None):
    """
    Display plots of each variable (position, velocity, pole angle, pole velocity) against time (/number of steps).
    If predicted states from a model are provided, the predicted dynamics are plotted alongside the true time evolutions.
    """
    positions = states[:,0]
    velocities = states[:,1]
    angles = states[:,2]
    pole_vels = states[:,3]
    if model:
        assert len(model_states) == len(states)
        pred_pos = model_states[:,0]
        pred_vel = model_states[:,1]
        pred_ang = model_states[:,2]
        pred_pol_vel = model_states[:,3]
    
    time = range(len(states))
    
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].plot(time, positions, label='True Cart Position')
    axs[1].plot(time, velocities, label='True Cart Velocity')
    axs[2].plot(time, angles, label='True Pole Angle')
    axs[3].plot(time, pole_vels, label='True Pole Velocity')
    if model:
        axs[0].plot(time, pred_pos, '--', label='Predicted Cart Position')
        axs[1].plot(time, pred_vel, '--', label='Predicted Cart Velocity')
        axs[2].plot(time, pred_ang, '--', label='Predicted Pole Angle')
        axs[3].plot(time, pred_pol_vel, '--', label='Predicted Pole Velocity')
    
    axs[0].set_ylabel('Cart Location')
    axs[1].set_ylabel('Cart Velocity')
    axs[2].set_ylabel('Pole Angle')
    axs[3].set_ylabel('Pole Velocity')
    
    for i in range(4):
            if model:
                axs[i].legend()
            axs[i].set_xlabel('Steps')
            axs[i].set_facecolor((0.1, 0.1, 0.1))
            axs[i].grid()
    plt.subplots_adjust(wspace=0.45)

def phase_portraits(states):
    positions = states[:,0]
    velocities = states[:,1]
    angles = states[:,2]
    pole_vels = states[:,3]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(positions, velocities)
    axs[0].set_ylabel('Cart Velocity')
    axs[0].set_xlabel('Cart Position')
    axs[1].plot(angles, pole_vels)
    axs[1].set_ylabel('Pole Velocity')
    axs[1].set_xlabel('Pole Angle')
    axs[0].set_facecolor((0.1, 0.1, 0.1))
    axs[0].grid()
    axs[1].set_facecolor((0.1, 0.1, 0.1))
    axs[1].grid()

def plot_y_1step(initial_state, ranges):
    """
    Vary the initial value of each variable one-by-one (i.e. keeping the other 3 constant, set by initial_state)
    One step of performAction for each variable value.
    Plot of Y for different initial values of each variable.
    Y is the system's state after one step of performAction.
    Four plots are produced.
    """
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for i in range(4): #Iterate over the four state variables
        positions = []
        velocities = []
        angles = []
        pole_vels = []
        rng = ranges[i]
        for val in ranges[i]: #Iterate over the specified range for the current variable (perform scan)
            initial_state_copy = initial_state.copy()
            initial_state_copy[i] = val
            y = simulate(initial_state=initial_state_copy)[1] # y = next state
            positions.append(y[0])
            velocities.append(y[1])
            angles.append(y[2])
            pole_vels.append(y[3])

        axs[i].plot(ranges[i], positions, 'b-', label='Cart Location') #Plot each variable as function of scan
        axs[i].plot(ranges[i], velocities, 'g-', label='Cart Velocity')
        axs[i].plot(ranges[i], angles, 'r-', label='Pole Angle')
        axs[i].plot(ranges[i], pole_vels, 'y-', label='Pole Velocity')
        axs[i].set_facecolor((0.1, 0.1, 0.1))
        axs[i].grid()
        axs[i].legend()
    axs[0].set_xlabel('Initial Value of Cart Location')
    axs[1].set_xlabel('Initial Value of Cart Velocity')
    axs[2].set_xlabel('Initial Value of Pole Angle')
    axs[3].set_xlabel('Initial Value of Pole Velocity')
    axs[0].set_ylabel('Y = X(1)')

def plot_y_diff(initial_state, ranges, linear_model=None, nonlinear_model=None, figsize=(14,10)):
    """
    Vary the initial value of each variable one-by-one (i.e. keeping the other 3 constant, set by initial_state)
    One step of performAction for each variable value.
    Plot of Y for different initial values of each variable.
    Y is the difference between the system's state after one performAction and the initial state!!!
    Four plots are produced.
    If a model is provided, the model's predictions are also plotted (on the same axes)
    """
    fig, ((axs1, axs2), (axs3, axs4)) = plt.subplots(2, 2, figsize=figsize)
    for i, axs in enumerate([axs1, axs2, axs3, axs4]):
        positions = []
        velocities = []
        angles = []
        pole_vels = []
        if linear_model:
            pred_pos = []
            pred_vel = []
            pred_ang = []
            pred_pol_vel = []
        if nonlinear_model:
            pred_pos2 = []
            pred_vel2 = []
            pred_ang2 =[]
            pred_pol_vel2 = []
        for val in ranges[i]:
            initial_state_copy = initial_state.copy()
            initial_state_copy[i] = val
            y = simulate(initial_state=initial_state_copy)[1] - np.array(initial_state_copy)
            positions.append(y[0])
            velocities.append(y[1])
            angles.append(y[2])
            pole_vels.append(y[3])
            if linear_model:
                y_pred = linear_model.predict([initial_state_copy])[0]
                pred_pos.append(y_pred[0])
                pred_vel.append(y_pred[1])
                pred_ang.append(y_pred[2])
                pred_pol_vel.append(y_pred[3])
            if nonlinear_model:
                alphas, kernel_centres, sigma = nonlinear_model[0], nonlinear_model[1], nonlinear_model[2]
                preds, preds_final = get_preds(np.array([initial_state_copy]), kernel_centres, sigma, alphas)
                pred_pos2.append(preds[0])
                pred_vel2.append(preds[1])
                pred_ang2.append(preds[2])
                pred_pol_vel2.append(preds[3])
        if i==0: #One legend is produced for whole subplot. This prevents same labels appearing in legend multiple times.
            label1, label2, label3, label4 = 'Cart Location', 'Cart Velocity', 'Pole Angle', 'Pole Velocity'
            label1x, label2x, label3x, label4x = 'Linear Predicted '+label1, 'Linear Predicted '+label2, 'Linear Predicted '+label3, 'Linear Predicted '+label4
            label1y, label2y, label3y, label4y = 'Non-Linear Predicted '+label1, 'Non-Linear Predicted '+label2, 'Non-Linear Predicted '+label3, 'Non-Linear Predicted '+label4
        else:
            label1 = label2 = label3 = label4 = None
            label1x = label2x = label3x = label4x = None
            label1y = label2y = label3y = label4y = None
        
        axs.plot(ranges[i], positions, 'b-', label=label1)
        axs.plot(ranges[i], velocities, 'g-', label=label2)
        axs.plot(ranges[i], angles, 'r-', label=label3)
        axs.plot(ranges[i], pole_vels, 'y-', label=label4)
        if linear_model:
            axs.plot(ranges[i], pred_pos, 'b--', label=label1x)
            axs.plot(ranges[i], pred_vel, 'g--', label=label2x)
            axs.plot(ranges[i], pred_ang, 'r--', label=label3x)
            axs.plot(ranges[i], pred_pol_vel, 'y--', label=label4x)
        if nonlinear_model:
            axs.plot(ranges[i], pred_pos2, 'b:', label=label1x)
            axs.plot(ranges[i], pred_vel2, 'g:', label=label2x)
            axs.plot(ranges[i], pred_ang2, 'r:', label=label3x)
            axs.plot(ranges[i], pred_pol_vel2, 'y:', label=label4x)
        axs.set_facecolor((0.1, 0.1, 0.1))
        axs.grid()
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=2)
    axs1.set_xlabel('Initial Value of Cart Location')
    axs2.set_xlabel('Initial Value of Cart Velocity')
    axs3.set_xlabel('Initial Value of Pole Angle')
    axs4.set_xlabel('Initial Value of Pole Velocity')
    axs1.set_ylabel('Y = X(1) - X(0)')

def plot_contours(initial_state, ranges):
    """
    Vary the initial values for TWO parameters, keeping the other two constant (set by initial_state)
    Contour plots of Y (= change in state after one step) as function of two scans.
    Four plots produced for each index pair (i.e. each scan).
    """
    for index in index_pairs: #obtain a pair of indices (e.g. [1, 3]) from a list of index pairs
        fig, axs = plt.subplots(1, 4, figsize=(14, 3))
        i, j = index[0], index[1]
        x = y = np.zeros((len(ranges[i]), len(ranges[j]), 4))
        for k in range(len(ranges[i])): #Scan over the two specified parameters
            for l in range(len(ranges[j])):
                val1, val2 = ranges[i][k], ranges[j][l]
                initial_state_copy = initial_state.copy()
                initial_state_copy[i] = val1
                initial_state_copy[j] = val2
                x[k,l] = initial_state_copy
                state = simulate(initial_state=initial_state_copy)[1]
                y[k,l] = state - np.array(initial_state_copy)
        axs[0].contourf(ranges[j], ranges[i], y[:,:,0], cmap='plasma') #Plot contours
        axs[1].contourf(ranges[j], ranges[i], y[:,:,1], cmap='plasma')
        axs[2].contourf(ranges[j], ranges[i], y[:,:,2], cmap='plasma')
        axs[3].contourf(ranges[j], ranges[i], y[:,:,3], cmap='plasma')
        axs[0].set_title('Cart Location')
        axs[1].set_title('Cart Velocity')
        axs[2].set_title('Pole Angle')
        axs[3].set_title('Pole Velocity')
        for k in range(4):
                axs[k].set_xlabel(labels[j])
                axs[k].set_ylabel(labels[i])
        plt.subplots_adjust(wspace=0.4)

def generate_data(n, train_prop=0.8, remap__angle=False, action_flag=False):
    """
    Generate n data points for training and testing a predictive model.
    The proportion of data set aside for training is set by train_prop (default = 80%)
    The input (x) is a random initial state.
    The output (y) is the change in state after a singular step.
    """
    x_stack = []
    y_stack = []
    for i in range(n):
        if action_flag:
            action = np.random.uniform(-20, 20)
        else:
            action = 0
        initial_state = [np.random.normal(loc=0, scale=1.5), np.random.uniform(-10, 10),
                        np.random.uniform(-np.pi, np.pi), np.random.uniform(-15, 15)] #Create random initial state
        x1 = simulate(initial_state=initial_state, remap__angle=remap__angle, action=action)[1] #Obtain state after one step
        y = x1 - np.array(initial_state) # y = change in state
        if action_flag:
            initial_state.append(action)
        x_stack.append(initial_state)
        y_stack.append(y)
    x_train, x_test = x_stack[:int(n*train_prop)], x_stack[int(n*train_prop):] #Split into proportion for training
    y_train, y_test = y_stack[:int(n*train_prop)], y_stack[int(n*train_prop):] #and testing
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

def create_initialStates(n, action_flag=False):
    """
    Create a set of initial states.
    First two states are already specificed (first is stable equilibrium)
    Next n states are randomly generated (within training set ranges)
    """
    initial_states = [[0, 0, np.pi, 0], [-0.169, 9.607, 2.557, -14.155], [0.738, -0.467, 3.068, 14.384]]
    for i in range(n):
        initial_states.append([np.random.normal(loc=0, scale=1.5), np.random.uniform(-10, 10),
                            np.random.uniform(-np.pi, np.pi), np.random.uniform(-15, 15)])

    return initial_states

def plot_ModelVsTrue_OverTime(steps, initial_states, model):
    """
    Plot true and predicted time evolutions (dynamics) of the cart-pole system
    for a range of given initial states.
    """
    for i in range(len(initial_states)):
        initial_state = initial_states[i]
        pred_states, initial_state_copy = initial_state.copy(), initial_state.copy()
        true_states = simulate(initial_state=initial_state, steps=steps, remap__angle=True) #Simulate for n steps
        for step in range(steps): #Predict n times using given model, starting from initial state
            initial_state_copy[2] = remap_angle(initial_state_copy[2])
            next_pred = model.predict([initial_state_copy])[0] + initial_state_copy
            next_pred[2] = remap_angle(next_pred[2])
            pred_states = np.vstack((pred_states, next_pred))
            initial_state_copy = next_pred
        #print(initial_state)
        display_plots(true_states, model=True, model_states=pred_states)

def get_kernel_centres(m, X):
    n = len(X)
    m_indices = []
    while len(m_indices) < m:
        num = np.random.randint(0, n)
        if num in m_indices:
            continue
        else:
            m_indices.append(num)
    kernel_centres = []
    for i in m_indices:
        kernel_centres.append(X[i])
    return kernel_centres

def kernel(X, X_prime, sigma):
    summ = 0
    for i in range(len(X)):
        if i != 2:
            val = ((X[i] - X_prime[i])**2)
        else:
            val = (np.sin((X[i]-X_prime[i])/2))**2
        summ += val/(2*sigma[i]**2)
    return np.exp(-summ)

def get_K_matrix(kernel_centres, X, sigma):
    n = len(X)
    m = len(kernel_centres)
    matrix = np.zeros((n, m))
    
    for i in range(n):
        for j in range(m):
            matrix[i, j] = kernel(X[i], kernel_centres[j], sigma=sigma)
    
    return matrix

def train_alpha(X, y, kernel_centres, sigma, lamda):
    m = len(kernel_centres)
    K_mm = get_K_matrix(kernel_centres, X[:m], sigma=sigma)
    K_nm = get_K_matrix(kernel_centres, X, sigma=sigma)
    alpha = np.linalg.lstsq((np.matmul(K_nm.T, K_nm) + lamda*K_mm), np.matmul(K_nm.T, y), rcond=None)[0]
    return alpha

def get_preds(x_test, kernel_centres, sigma, alphas):
    K_nm_test = get_K_matrix(kernel_centres, x_test, sigma)
    preds = []
    for i in range(4):
        preds.append(np.matmul(K_nm_test, alphas[i]))
    preds_final = []
    for i, pred in enumerate(preds):
        pred_final = np.add(pred, x_test[:,i])
        preds_final.append(pred_final)
    return preds, preds_final

def plot_predicted_against_true(preds_final, y_test_final):
    fig, ((axs1, axs2), (axs3, axs4)) = plt.subplots(2, 2, figsize=(16, 10))
    for i, axs in enumerate([axs1, axs2, axs3, axs4]):
        c = np.abs(preds_final[i] - y_test_final[:,i])
        axs.scatter(y_test_final[:,i], preds_final[i], c=c, cmap='plasma', s=8)
        axs.set_facecolor((0.1, 0.1, 0.1))
        axs.grid()

    axs1.set_xlabel('True Cart Location')
    axs1.set_ylabel('Predicted Cart Location')

    axs2.set_xlabel('True Cart Velocity')
    axs2.set_ylabel('Predicted Cart Velocity')

    axs3.set_xlabel('True Pole Angle')
    axs3.set_ylabel('Predicted Pole Angle')

    axs4.set_xlabel('True Pole Velocity')
    axs4.set_ylabel('Predicted Pole Velocity')

    plt.subplots_adjust(wspace=0.2)

def find_best_lambda(lambdas):
    labels = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Velocity']
    n=1000
    x_train, y_train, x_test, y_test = generate_data(n, train_prop=0.8)
    sigma = [10, 16, 0.5, 11]
    m = 100
    kernel_centres = get_kernel_centres(m, x_train)
    K_nm_test = get_K_matrix(kernel_centres, x_test, sigma)
    errors = []
    for l in lambdas:
        alphas = []
        errors_for_given_l = []
        for i in range(4):
            alpha = train_alpha(X=x_train, y=y_train[:,i], kernel_centres=kernel_centres, sigma=sigma, lamda=l)
            pred = np.matmul(K_nm_test, alpha)
            errors_for_given_l.append(mse(y_test[:,i], pred))  
        errors.append(errors_for_given_l)
    errors = np.array(errors)
    fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    for i in range(4):
        axs.plot(lambdas, errors[:,i], label=labels[i])
    axs.set_xscale('log')
    axs.set_facecolor((0.1, 0.1, 0.1))
    axs.grid()
    axs.set_title('Mean Squared Error in Predicted Change of State')
    axs.set_ylabel('MSE')
    axs.set_xlabel('Lambda')
    axs.legend()

def find_best_N(Ns, x_test, y_test, m):
    labels = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Velocity']
    sigma = [10, 16, 0.5, 11]
    errors = []
    for n in Ns:
        n = int(n)
        x_train, y_train, ignore1, ignore2 = generate_data(n, train_prop=1)
        kernel_centres = get_kernel_centres(m, x_train)
        K_nm_test = get_K_matrix(kernel_centres, x_test, sigma)
        alphas = []
        errors_for_given_n = []
        for i in range(4):
            alpha = train_alpha(X=x_train, y=y_train[:,i], kernel_centres=kernel_centres, sigma=sigma, lamda=1e-5)
            pred = np.matmul(K_nm_test, alpha)
            errors_for_given_n.append(mse(y_test[:,i], pred))  
        errors.append(errors_for_given_n)
    errors = np.array(errors)
    fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    for i in range(4):
        axs.plot(Ns, errors[:,i], label=labels[i])
    axs.set_facecolor((0.1, 0.1, 0.1))
    axs.grid()
    axs.set_title('Mean Squared Error in Predicted Change of State')
    axs.set_ylabel('MSE')
    axs.set_xlabel('N')
    axs.legend()

def find_best_N2(Ns, x_test, y_test):
    labels = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Velocity']
    sigma = [10, 16, 0.5, 11]
    errors = []
    for n in Ns:
        n = int(n)
        m = int(0.1*n)
        x_train, y_train, ignore1, ignore2 = generate_data(n, train_prop=1)
        kernel_centres = get_kernel_centres(m, x_train)
        K_nm_test = get_K_matrix(kernel_centres, x_test, sigma)
        alphas = []
        errors_for_given_n = []
        for i in range(4):
            alpha = train_alpha(X=x_train, y=y_train[:,i], kernel_centres=kernel_centres, sigma=sigma, lamda=1e-5)
            pred = np.matmul(K_nm_test, alpha)
            errors_for_given_n.append(mse(y_test[:,i], pred))  
        errors.append(errors_for_given_n)
    errors = np.array(errors)
    fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    for i in range(4):
        axs.plot(Ns, errors[:,i], label=labels[i])
    axs.set_facecolor((0.1, 0.1, 0.1))
    axs.grid()
    axs.set_title('Mean Squared Error in Predicted Change of State')
    axs.set_ylabel('MSE')
    axs.set_xlabel('N')
    axs.legend()

def find_best_M(Ms, n, sigma, lam):
    labels = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Velocity']
    x_train, y_train, x_test, y_test = generate_data(n, train_prop=0.8)
    errors = []
    for m in Ms:
        kernel_centres = get_kernel_centres(m, x_train)
        K_nm_test = get_K_matrix(kernel_centres, x_test, sigma)
        alphas = []
        errors_for_given_m = []
        for i in range(4):
            alpha = train_alpha(X=x_train, y=y_train[:,i], kernel_centres=kernel_centres, sigma=sigma, lamda=lam)
            pred = np.matmul(K_nm_test, alpha)
            errors_for_given_m.append(mse(y_test[:,i], pred))  
        errors.append(errors_for_given_m)
    errors = np.array(errors)
    fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    for i in range(4):
        axs.plot(Ms, errors[:,i], label=labels[i])
    axs.set_facecolor((0.1, 0.1, 0.1))
    axs.grid()
    axs.set_title('Mean Squared Error in Predicted Change of State')
    axs.set_ylabel('MSE')
    axs.set_xlabel('M (Number of Kernel Centres)')
    axs.legend()

def plot_model_contours(initial_state, ranges, nonlinear_model):
    """
    Vary the initial values for TWO parameters, keeping the other two constant (set by initial_state)
    Contour plots of Y (= change in state after one step) as function of two scans.
    Four plots produced for each index pair (i.e. each scan).
    """
    for index in index_pairs: #obtain a pair of indices (e.g. [1, 3]) from a list of index pairs
        fig, axs = plt.subplots(1, 4, figsize=(14, 3))
        i, j = index[0], index[1]
        x = y = np.zeros((len(ranges[i]), len(ranges[j]), 4))
        for k in range(len(ranges[i])): #Scan over the two specified parameters
            for l in range(len(ranges[j])):
                val1, val2 = ranges[i][k], ranges[j][l]
                initial_state_copy = initial_state.copy()
                initial_state_copy[i] = val1
                initial_state_copy[j] = val2
                x[k,l] = initial_state_copy
                alphas, kernel_centres, sigma = nonlinear_model[0], nonlinear_model[1], nonlinear_model[2]
                preds, preds_final = get_preds(np.array([initial_state_copy]), kernel_centres, sigma, alphas)
                y[k,l] = preds
        axs[0].contourf(ranges[j], ranges[i], y[:,:,0], cmap='plasma') #Plot contours
        axs[1].contourf(ranges[j], ranges[i], y[:,:,1], cmap='plasma')
        axs[2].contourf(ranges[j], ranges[i], y[:,:,2], cmap='plasma')
        axs[3].contourf(ranges[j], ranges[i], y[:,:,3], cmap='plasma')
        axs[0].set_title('Cart Location')
        axs[1].set_title('Cart Velocity')
        axs[2].set_title('Pole Angle')
        axs[3].set_title('Pole Velocity')
        for k in range(4):
                axs[k].set_xlabel(labels[j])
                axs[k].set_ylabel(labels[i])
        plt.subplots_adjust(wspace=0.4)

def plot_ModelVsTrue_OverTime2(steps, initial_states, nonlinear_model, action=0):
    """
    Plot true and predicted time evolutions (dynamics) of the cart-pole system
    for a range of given initial states.
    """
    alphas, kernel_centres, sigma = nonlinear_model[0], nonlinear_model[1], nonlinear_model[2]
    for i in range(len(initial_states)):
        initial_state = initial_states[i]
        pred_states, initial_state_copy = initial_state.copy(), initial_state.copy()
        true_states = simulate(initial_state=initial_state, steps=steps, remap__angle=True, action=action) #Simulate for n steps
        for _ in range(steps): #Predict n times using given model, starting from initial state
            initial_state_copy[2] = remap_angle(initial_state_copy[2])
            if action != 0:
                initial_state_copy.append(action)
            preds, preds_final = get_preds(np.array([initial_state_copy]), kernel_centres, sigma, alphas)
            next_pred = [0, 0, 0, 0]
            for i in range(4):
                next_pred[i] = preds_final[i][0]
            next_pred[2] = remap_angle(next_pred[2])
            pred_states = np.vstack((pred_states, next_pred))
            initial_state_copy = next_pred
        display_plots(true_states, model=True, model_states=pred_states)

def plot_y_diff2(initial_state, initial_force, ranges, linear_model=None, nonlinear_model=None, figsize=(14,10)):
    """
    Vary the initial value of each variable one-by-one (i.e. keeping the other 3 constant, set by initial_state)
    One step of performAction for each variable value.
    Plot of Y for different initial values of each variable.
    Y is the difference between the system's state after one performAction and the initial state!!!
    Four plots are produced.
    If a model is provided, the model's predictions are also plotted (on the same axes)
    """
    fig, ((axs1, axs2), (axs3, axs4), (axs5, axs6)) = plt.subplots(3, 2, figsize=figsize)
    for i, axs in enumerate([axs1, axs2, axs3, axs4, axs5]):
        positions = []
        velocities = []
        angles = []
        pole_vels = []
        if linear_model:
            pred_pos = []
            pred_vel = []
            pred_ang = []
            pred_pol_vel = []
        if nonlinear_model:
            pred_pos2 = []
            pred_vel2 = []
            pred_ang2 =[]
            pred_pol_vel2 = []
        for val in ranges[i]:
            initial_state_copy = initial_state.copy()
            if i != 4:
                initial_state_copy[i] = val
                action = initial_force
            else:
                action = val
            y = simulate(initial_state=initial_state_copy, action=action)[1] - np.array(initial_state_copy)
            positions.append(y[0])
            velocities.append(y[1])
            angles.append(y[2])
            pole_vels.append(y[3])
            if linear_model:
                y_pred = linear_model.predict([initial_state_copy])[0]
                pred_pos.append(y_pred[0])
                pred_vel.append(y_pred[1])
                pred_ang.append(y_pred[2])
                pred_pol_vel.append(y_pred[3])
            if nonlinear_model:
                alphas, kernel_centres, sigma = nonlinear_model[0], nonlinear_model[1], nonlinear_model[2]
                initial_state_copy.append(action)
                preds, preds_final = get_preds(np.array([initial_state_copy]), kernel_centres, sigma, alphas)
                pred_pos2.append(preds[0])
                pred_vel2.append(preds[1])
                pred_ang2.append(preds[2])
                pred_pol_vel2.append(preds[3])
        if i==0: #One legend is produced for whole subplot. This prevents same labels appearing in legend multiple times.
            label1, label2, label3, label4 = 'Cart Location', 'Cart Velocity', 'Pole Angle', 'Pole Velocity'
            label1x, label2x, label3x, label4x = 'Linear Predicted '+label1, 'Linear Predicted '+label2, 'Linear Predicted '+label3, 'Linear Predicted '+label4
            label1y, label2y, label3y, label4y = 'Non-Linear Predicted '+label1, 'Non-Linear Predicted '+label2, 'Non-Linear Predicted '+label3, 'Non-Linear Predicted '+label4
        else:
            label1 = label2 = label3 = label4 = None
            label1x = label2x = label3x = label4x = None
            label1y = label2y = label3y = label4y = None
        
        axs.plot(ranges[i], positions, 'b-', label=label1)
        axs.plot(ranges[i], velocities, 'g-', label=label2)
        axs.plot(ranges[i], angles, 'r-', label=label3)
        axs.plot(ranges[i], pole_vels, 'y-', label=label4)
        if linear_model:
            axs.plot(ranges[i], pred_pos, 'b--', label=label1x)
            axs.plot(ranges[i], pred_vel, 'g--', label=label2x)
            axs.plot(ranges[i], pred_ang, 'r--', label=label3x)
            axs.plot(ranges[i], pred_pol_vel, 'y--', label=label4x)
        if nonlinear_model:
            axs.plot(ranges[i], pred_pos2, 'b:', label=label1y)
            axs.plot(ranges[i], pred_vel2, 'g:', label=label2y)
            axs.plot(ranges[i], pred_ang2, 'r:', label=label3y)
            axs.plot(ranges[i], pred_pol_vel2, 'y:', label=label4y)
        axs.set_facecolor((0.1, 0.1, 0.1))
        axs.grid()
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=2)
    axs1.set_xlabel('Initial Value of Cart Location')
    axs2.set_xlabel('Initial Value of Cart Velocity')
    axs3.set_xlabel('Initial Value of Pole Angle')
    axs4.set_xlabel('Initial Value of Pole Velocity')
    axs.set_xlabel('Initial Value of Force')
    axs1.set_ylabel('Y = X(1) - X(0)')
    axs6.set_visible(False)