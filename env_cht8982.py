"""Import required libraries for Custom environment Chakravyuh."""

import gymnasium
import numpy as np
import pygame
from pathlib import Path
import random

class ChakravyuhEnv(gymnasium.Env):
    """Custom environment representing the Chakravyuh maze."""

    def __init__(self):
        """
        Initialize the Chakravyuh environment.
        Defines grid size, agent's position, goal position, obstacles, and their powers.
        """        
        self.grid_size = (7, 7)
        self.action_space = gymnasium.spaces.Discrete(4)
        self.agent_position = np.array([0, 0])
        self.goal_position = np.array([3, 3])
        
        self.obstacle_positions = [
            {"name": "Karn", "position": np.array([2, 1])}, 
            {"name": "Balram", "position": np.array([1, 3])}, 
            {"name": "Shakuni", "position": np.array([3, 1])}, 
            {"name": "Ashwatthama", "position": np.array([1, 5])}, 
            {"name": "Drona", "position": np.array([5, 1])}, 
            {"name": "Bhishma", "position": np.array([5, 3])},
            {"name": "Kritavarma", "position": np.array([5, 5])}, 
            {"name": "Jayadratha", "position": np.array([3, 5])},
            {"name": "Kripa", "position": np.array([1, 2])},
            {"name": "Dushasan", "position": np.array([4, 4])},
            {"name": "Vikarna", "position": np.array([2, 4])}
        ]
        
        self._max_episode_steps = 1000
        self._current_step = 0
        self.reward = 0
        self.cumulative_reward = 0
        self.primary_obstacle_reward = -60
        self.goal_reward = 400
        self.living_reward = -1
        self.use_motivation = True #self.use_motivation = True  # Set this flag to control motivation reward conditions
        self.motivate_rate = 0.2
        self.motivate_decay = 0.2
        self.motivate_positions = [
                                    (np.array([3, 2]), 0), 
                                    (np.array([4, 3]), 0),
                                    (np.array([4, 1]), 1), 
                                    (np.array([4, 2]), 1), 
                                    (np.array([5, 2]), 1),
                                    (np.array([4, 0]), 2), 
                                    (np.array([6, 2]), 2)
                                    ]
        self.maximum_negative_reward_limit = -300
        self.reverse_action = False  # Reverse action flag
        self.reverse_hit_count = 0  # Count of reverse hits
        
        self.obstacles = {
            "Shakuni": {"power": "reverse"},
            "Kritavarma": {"penalty": -10, "power": "penalty"},
            "Ashwatthama": {"teleport_random": True, "power": "teleport_random"},
            "Bhishma": {"effect": 4, "power": "freeze"},
            "Drona": {"teleport_position": np.array([0, 0]), "power": "teleport"},
            "Karn": {"health_reduction": 1, "power": "health_reduction"},
            "Balram": {"score_multiplier": 4, "power": "score_multiplier"},
            "Jayadratha": {"restrict_left": (np.array([3, 5]), np.array([3, 3])), "power": "restrict_left"},
            "Kripa": {"effect": 2, "power": "freeze"},
            "Dushasan": {"teleport_position": np.array([6, 6]), "power": "teleport"},
            "Vikarna": {"power": "game_over"}
        }

        self.health = 3
        self.score_multiplier = 1.0
        pygame.font.init()
        self.font = pygame.font.SysFont(None, 24)
        self.agent_path = []  # List to keep track of agent's path

    def reset(self, random_position=True):
        """
        Reset the environment to the initial state.

        Returns:
            np.array: The initial position of the agent.
        """ 
        if random_position:
            # Generate a random position not in obstacle positions and not the goal position
            random_state = np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])
            while any((random_state == tuple(obs["position"])) for obs in self.obstacle_positions) or (random_state == tuple(self.goal_position)):
                random_state = np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])
            self.agent_position = np.array(random_state)
        else:
            self.agent_position = np.array([0, 0])
           

        self._current_step = 0
        self.cumulative_reward = 0
        self.reward = 0
        self.freeze_steps = 0
        self.health = 3
        self.score_multiplier = 1.0
        self.restricted_left = False
        self.agent_path = [self.agent_position.copy()]  # Reset the path and add initial position
        self.reverse_action = False
        self.reverse_hit_count = 0
        return self.agent_position
    

    def step(self, action):
        """
        Execute a step in the environment based on the given action.

        Args:
            action (int): The action to be taken by the agent.

        Returns:
            np.array: New position of the agent.
            float: Cumulative reward.
            bool: Whether the episode is done.
            dict: Additional information.
        """
        initial_position = self.agent_position.copy()
        first_time = self._current_step == 0

        if self.reverse_action:
            action = self._reverse_action(action)

        # Handle freeze steps
        if self.freeze_steps > 0:
            print(f"Agent is frozen for {self.freeze_steps} more steps.")
            self.freeze_steps -= 1
            self._current_step += 1
            return self.agent_position, self.reward, False, {}

        # Handle restricted left movement by Jayadratha
        if self.restricted_left and action == 2 and (self.agent_position[0] == 3 and 3 <= self.agent_position[1] <= 5):
            print("Left movement restricted by Jayadratha.")
            self._current_step += 1
            return self.agent_position, self.reward, False, {}

        # Update the agent's position based on the action
        action_mapping = {0: np.array([-1, 0]), 1: np.array([1, 0]), 2: np.array([0, -1]), 3: np.array([0, 1])}
        new_position = self.agent_position + action_mapping[action]

        if not self._is_within_bounds(new_position):
            new_position = self.agent_position  # Agent stays in the same position if out of bounds
        else:
            obstacle = next((obs for obs in self.obstacle_positions if (new_position == obs["position"]).all()), None)
            if obstacle:
                obstacle_name = obstacle["name"]
                obstacle_power = self.obstacles[obstacle_name]["power"]
                result = self._handle_obstacle(obstacle_name, obstacle_power, new_position)
                if result is not None:
                    return result

        self.agent_position = new_position
        self.agent_path.append(self.agent_position.copy())  # Append new position to the path
        self.reward = self._calculate_reward()
        self.cumulative_reward += self.reward
        done = (self.agent_position == self.goal_position).all()
        self._current_step += 1

        # Check for end conditions
        if self.cumulative_reward <= self.maximum_negative_reward_limit:
            print(f"Cumulative reward reached maximum limit {self.maximum_negative_reward_limit}.")
            done = True
        elif self._current_step >= self._max_episode_steps:
            done = True
            if not (self.agent_position == self.goal_position).all():
                print("Abhimanyu is trapped in the Chakravyuh.")
        elif done:
            print("Abhimanyu has reached Duryodhan.")
            print(f"New State: {self.agent_position}, reward: {self.reward}, Cumulative Reward: {self.cumulative_reward}, Distance to Goal: [0 0], Required no. of actions: 0")

        required_action = np.sum(np.abs(self.agent_position - self.goal_position))
        distance_to_goal = np.array(self.agent_position - self.goal_position)

        if first_time:
            print(f"Initial State: {initial_position}")
            print(f"New State: {self.agent_position}, reward: {self.reward}, Cumulative Reward: {self.cumulative_reward}, Distance to Goal: {distance_to_goal}, Required no. of actions: {required_action}")
        elif not done:  # Only print the new state if the game is not done
            print(f"New State: {self.agent_position}, reward: {self.reward}, Cumulative Reward: {self.cumulative_reward}, Distance to Goal: {distance_to_goal}, Required no. of actions: {required_action}")

        return self.agent_position, self.reward, done, {}


    def render(self, screen):
        """
        Render the environment using images of the agent, goal, and obstacles.

        Args:
            screen (pygame.Surface): The Pygame surface to draw the environment on.
        """
        background = pygame.image.load(str(Path("images") / "chkravyuha.jpeg"))
        background = pygame.transform.scale(background, (self.grid_size[1] * 100, self.grid_size[0] * 100))
        screen.blit(background, (0, 0))

        agent_img = pygame.image.load(str(Path("images") / "abhimanyu.jpeg"))
        agent_img = pygame.transform.scale(agent_img, (100, 100))
        goal_img = pygame.image.load(str(Path("images") / "duryodhan.jpeg"))
        goal_img = pygame.transform.scale(goal_img, (100, 100))

        obstacle_imgs = {}
        for obstacle in self.obstacle_positions:
            obstacle_imgs[obstacle["name"]] = pygame.image.load(str(Path("images") / f'{obstacle["name"].lower()}.jpeg'))
            obstacle_imgs[obstacle["name"]] = pygame.transform.scale(obstacle_imgs[obstacle["name"]], (100, 100))

        # Draw the path of the agent
        if len(self.agent_path) > 1:
            for i in range(len(self.agent_path) - 1):
                start_pos = (self.agent_path[i][1] * 100 + 50, self.agent_path[i][0] * 100 + 50)
                end_pos = (self.agent_path[i + 1][1] * 100 + 50, self.agent_path[i + 1][0] * 100 + 50)
                pygame.draw.line(screen, (0, 0, 255), start_pos, end_pos, 5)  # Blue line with thickness 5

        screen.blit(agent_img, (self.agent_position[1] * 100, self.agent_position[0] * 100))
        screen.blit(goal_img, (self.goal_position[1] * 100, self.goal_position[0] * 100))

        for obstacle in self.obstacle_positions:
            screen.blit(obstacle_imgs[obstacle["name"]], (obstacle["position"][1] * 100, obstacle["position"][0] * 100))
            # Render the name of the obstacle
            text_surface = self.font.render(obstacle["name"], True, (255, 0, 0))  # red color
            screen.blit(text_surface, (obstacle["position"][1] * 100 + 5, obstacle["position"][0] * 100 + 85))

        # Render the name of the agent and the goal
        agent_text_surface = self.font.render("Abhimanyu", True, (0, 0, 255))  # Blue color
        screen.blit(agent_text_surface, (self.agent_position[1] * 100 + 5, self.agent_position[0] * 100 + 85))

        goal_text_surface = self.font.render("Duryodhan", True, (0, 255, 0))  # Green color
        screen.blit(goal_text_surface, (self.goal_position[1] * 100 + 5, self.goal_position[0] * 100 + 85))

    def _reverse_action(self, action):
        reverse_mapping = {0: 1, 1: 0, 2: 3, 3: 2}
        return reverse_mapping[action]

    def _is_within_bounds(self, position):
        """
        Check if a position is within the grid bounds.

        Args:
            position (np.array): The position to check.

        Returns:
            bool: True if within bounds, False otherwise.
        """
        return (0 <= position[0] < self.grid_size[0]) and (0 <= position[1] < self.grid_size[1])
    

    

    def _handle_obstacle(self, obstacle_name, obstacle_power, new_position):
        """Handle interactions with obstacles.
        
        Args:
            obstacle_name (str): Name of the obstacle.
            obstacle_power (str): Power of the obstacle.
            new_position (np.array): The new position of the agent.
        
        Returns:
            Tuple: Updated agent position, cumulative reward, done status, and info dictionary.
        """        
        if obstacle_power == "penalty":
            penalty = self.obstacles[obstacle_name]["penalty"]
            print(f"Abhimanyu has been additionally penalized by {obstacle_name} by {penalty} points.")

        elif obstacle_power == "freeze":
            self.freeze_steps = self.obstacles[obstacle_name]["effect"]
            print(f"Abhimanyu has been frozen by {obstacle_name} for {self.freeze_steps} steps.")
        # elif obstacle_power == "health_reduction":
        #     penalty = (self.maximum_negative_reward_limit-self.cumulative_reward) / self.health if self.health > 0 else self.maximum_negative_reward_limit
        #     self.health -= self.obstacles[obstacle_name]["health_reduction"]
        #     print(f"Abhimanyu's health power reduced by {obstacle_name}. Current health: {self.health}")            
        #     self.cumulative_reward += penalty
        #     print(f"Karn's additional reward penalty applied: {penalty}.")
        elif obstacle_power == "teleport":
            new_position[:] = self.obstacles[obstacle_name]["teleport_position"]
            print(f"Abhimanyu has been teleported by {obstacle_name} to position {new_position}.")
            #self.cumulative_reward -= 5 * self.score_multiplier  # Apply multiplier for teleport obstacles
        elif obstacle_power == "score_multiplier":
            self.score_multiplier = self.obstacles[obstacle_name]["score_multiplier"]
            print(f"Abhimanyu's score multiplier changed by {obstacle_name} to {self.score_multiplier}.")
        elif obstacle_power == "restrict_left":
            start, end = self.obstacles[obstacle_name]["restrict_left"]
            if start[0] == end[0] and start[0] == self.agent_position[0]:
                self.restricted_left = True
                print(f"Abhimanyu's left movement restricted by {obstacle_name} from position {start} to {end}.")
        elif obstacle_power == "teleport_random":
            new_position[:] = np.array([random.randint(0, 6), random.randint(0, 6)])
            print(f"Abhimanyu has been randomly teleported by {obstacle_name} to position {new_position}.")
            #self.cumulative_reward -= 5 * self.score_multiplier  # Apply multiplier for teleport obstacles
        # elif obstacle_power == "game_over":
        #     vikarna_penalty = (self.maximum_negative_reward_limit-self.cumulative_reward)
        #     self.cumulative_reward += vikarna_penalty
        #     print(f"Abhimanyu has encountered by {obstacle_name}'s additional penalty {vikarna_penalty}, Cumulative Reward: {self.cumulative_reward}. Game over.")
        #     return self.agent_position, self.cumulative_reward, True, {}
        elif obstacle_power == "reverse":
            self.reverse_action = not self.reverse_action
            self.reverse_hit_count += 1
            print(f"Abhimanyu has been {obstacle_power} by {obstacle_name}")
        

    def _calculate_reward(self):
        """
        Calculate reward based on the agent's current position.

        Returns:
            float: The calculated reward.
        """
        #self.use_motivation = True  # Set this flag to control motivation reward conditions
      

        
        # Check if the agent is at the goal position
        if (self.agent_position == self.goal_position).all():
            self.reward  = self.goal_reward
            return self.reward
        
        # Apply motivational rewards if the flag is set
        if self.use_motivation:
            for pos, level in self.motivate_positions:
                if (self.agent_position == pos).all():
                    decay_factor = self.motivate_decay ** level #level is 0=closest, 1=closer,2=close
                    self.reward = (self.motivate_rate * decay_factor) * self.goal_reward
                    print(self.reward)
                    return self.reward
  
        # Check if the agent has encountered an obstacle
        for obs in self.obstacle_positions:
            if (self.agent_position == obs["position"]).all():             
                base_penalty = self.primary_obstacle_reward * self.score_multiplier
                character_penalty = 0
                obstacle_name = obs["name"]
                if obstacle_name in self.obstacles:
                    obstacle = self.obstacles[obstacle_name]
                    if "penalty" in obstacle:
                        character_penalty += obstacle["penalty"] * self.score_multiplier
                    elif obstacle_name == "Karn":
                        karn_penalty = (self.maximum_negative_reward_limit-self.cumulative_reward-base_penalty) / self.health if self.health > 0 else self.maximum_negative_reward_limit
                        self.health -= self.obstacles[obstacle_name]["health_reduction"]
                        print(f"Abhimanyu's health power reduced by {obstacle_name}. Current health: {self.health}")            
                        print(f"Karn's additional reward penalty applied: {karn_penalty}.")
                        character_penalty += karn_penalty
                    elif obstacle_name == "Vikarna":
                        vikarna_penalty = (self.maximum_negative_reward_limit-self.cumulative_reward-base_penalty)
                        character_penalty += vikarna_penalty
                        print(f"Abhimanyu has encountered by {obstacle_name}'s additional penalty {vikarna_penalty}, Cumulative Reward: {self.cumulative_reward}. Game over.")



                obstacle_reward = base_penalty + character_penalty
                print(f'base_penalty{base_penalty} + character_penalty {character_penalty} = {obstacle_reward} obstacle_reward')
                self.reward = obstacle_reward

                return self.reward #obstacle_reward
            

        # Default reward for living   
        else:
            self.reward  = self.living_reward
            return self.reward

   
# Create Chakravyuh environment instance
env = ChakravyuhEnv()

# Pygame for rendering
pygame.init()

# Set up Pygame display
WINDOW_SIZE = (700, 700)
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Chakravyuh Visualization")
clock = pygame.time.Clock()

# # Load and play the background sound
# pygame.mixer.init()
# pygame.mixer.music.load('sound.mp3')  
# pygame.mixer.music.play(-1)  # Loop the sound indefinitely



# # Reset the environment to the initial state
# env.reset()

# # Main loop
# running = True
# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False
#         elif event.type == pygame.KEYDOWN:
#             if event.key == pygame.K_UP:
#                 action = 0  # Up
#             elif event.key == pygame.K_DOWN:
#                 action = 1  # Down
#             elif event.key == pygame.K_LEFT:
#                 action = 2  # Left
#             elif event.key == pygame.K_RIGHT:
#                 action = 3  # Right
#             else:
#                 continue  # Ignore other keys

#             # Perform the action in the environment
#             observation, cumulative_reward, done, _ = env.step(action)

#             # Check if episode is done
#             if done:
#                 running = False

#     # Render the environment
#     env.render(screen)

#     pygame.display.flip()
#     clock.tick(60)

# pygame.quit()
# pygame.mixer.quit()  # Properly quit the mixer
