import numpy as np

### The random walker (bug) can take any of the possible steps in the 2d euclidean space
possible_steps = np.array([[-10,0], [10,0], [0,10], [0,-10]])

###Function to simulate a 2D random walk
def walk_2d(condition, steps_per_sample=100):
    #Starting at origin
    current_position = np.array([0,0])
    #Count the number of batches of steps taken
    counter = -1
    #Boolean flag to keep record of whether random walker has been absorbed
    absorbed = False
    #Run a loop until random walker has not been absorbed
    while(not absorbed):
        #Increase step batch counter
        counter = counter + 1
        #Randomly sample steps for the random walker
        step_choices = np.random.choice([0,1,2,3],size=steps_per_sample)
        #2D array where the nth row is the position of the random walker after the nth step in this random sample of steps
        positions = possible_steps[step_choices].cumsum(axis=0)+current_position
        #Position after the last step in this random sample of steps
        current_position = positions[-1]
        x = positions[:,0]
        y = positions[:,1]
        #Checking for absorption
        position_conditions = condition(x,y)
        if(position_conditions.sum()>0):
            #If absorbed, return the step in which the random walker was absorbed
            return (counter*steps_per_sample)+position_conditions.argmax()+1

###Handler function which keeps track of convergence of the random walk in a very crude manner
def run_simulation(condition, steps_per_sample=100, n_iter=1000, n_batches=100, epsilon=1e-3):
    #Array to store means from all batches
    means = []
    #Mean value of all simulations before including latest batch of simulations
    prev_mean = float('nan')
    #Mean value of all simulations after including latest batch of simulations
    current_mean = float('nan')
    #Boolean flag to check if Monte Carlo simulation has converged
    converged = False
    for batch_n in range(n_batches):
        #Absroption Time from samples in current batch
        absroption_times = np.array([walk_2d(condition, steps_per_sample) for _ in range(n_iter)])
        #Appending the latest mean value
        means.append(absroption_times.mean())
        #Updating mean values
        prev_mean = current_mean
        current_mean = sum(means)/len(means)
        #Check for convergence. If converged stop the loop.
        if(abs(current_mean-prev_mean)<epsilon):
            converged = True
            break
    if converged:
        print(f"Converged in {batch_n} batches : Expected absorption time (as int {int(np.round(current_mean))}), (as float {current_mean})")
    else:
        print(f"Simulation did not converge")
    return None

#Condition for absorption state where the random walker can find food in part 1
def condition1(x,y):
    return np.logical_or(np.abs(x)>=20, np.abs(y)>=20)

#Condition for absorption state where the random walker can find food in part 2
def condition2(x,y):
    return (x+y)>=10
   
#Condition for absorption state where the random walker can find food in part 3
def condition3(x,y):
    return pow(((x-2.5)/30),2)+pow(((y-2.5)/40),2)>=1

print("Q1 : ", end="")
run_simulation(condition1)
print("Q3 : ", end="")
run_simulation(condition3)
#print("Q2 : ", end="")
#run_simulation(condition2, steps_per_sample=1000, n_batches=10)

