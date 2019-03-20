import numpy as np

def rot2d(angle):
    return np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle),np.cos(angle)]])

def getFrustrumPoints(bot_location,bot_angle,min_d,max_d):
    fov = 69.4
    fov_rad = np.deg2rad(fov)
    
    frustrum_t0 = bot_location+np.array([min_d,0])@rot2d(-fov_rad/2-bot_angle);
    frustrum_b0 = bot_location+np.array([min_d,0])@rot2d(fov_rad/2-bot_angle);
    frustrum_t1 = bot_location+np.array([max_d,0])@rot2d(-fov_rad/2-bot_angle);
    frustrum_b1 = bot_location+np.array([max_d,0])@rot2d(fov_rad/2-bot_angle);
    return np.array([frustrum_t0, frustrum_t1, frustrum_b1, frustrum_b0])

