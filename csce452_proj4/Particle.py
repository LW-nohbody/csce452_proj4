from sklearn.neighbors import KernelDensity
import numpy as np

class Particle():

    dark_data = np.array([135, 128, 150, 123, 145, 140, 123, 140, 148, 125, 
                          149, 137, 135, 148, 129, 132, 144, 130, 124, 134, 
                          142, 122, 129, 138, 148, 130, 143, 148, 143, 141, 
                          148, 145, 135, 133, 141, 130, 129, 134, 142, 123, 
                          131, 147, 139, 123, 124, 128, 140, 129, 139, 132, 
                          143, 150, 135, 137, 126, 128, 130, 135, 150, 137, 
                          148, 133, 138, 126, 148, 128, 152, 122, 138, 123, 
                          139, 138, 129, 151, 143, 127, 139, 127, 149, 128, 
                          129, 140, 136, 131, 136, 151, 122, 132, 150, 130, 
                          122, 126, 137, 125, 140, 127, 148, 141, 133, 132, 
                          149, 152, 139, 126, 123, 149, 151, 142, 143, 143, 
                          146, 123, 135, 131, 148, 142, 137, 148, 140, 148, 
                          127, 124, 144, 140, 132, 140, 123, 122, 147, 138, 
                          145, 138, 130, 144, 126, 145, 149, 140, 148, 149, 
                          122, 139, 130, 125, 137, 122, 131, 128, 128, 146, 
                          138, 143, 146, 136, 127, 122, 146, 129, 129, 129, 
                          129, 136, 149
                        ])
    light_data = np.array([106, 121, 131, 105, 107, 121, 110, 127, 115, 131, 
                           111, 107, 121, 111, 118, 115, 107, 121, 129, 102, 
                           131, 114, 128, 123, 122, 128, 129, 105, 116, 129, 
                           127, 102, 104, 114, 132, 131, 107, 113, 112, 115, 
                           110, 132, 111, 126, 114, 127, 105, 126, 132, 113, 
                           109, 107, 107, 120, 116, 118, 126, 111, 118, 110, 
                           129, 102, 106, 126, 129, 118, 115, 117, 104, 123, 
                           128, 107, 113, 111, 104, 115, 107, 106, 128, 121, 
                           126, 102, 121, 115, 121, 126, 129, 128, 102, 118, 
                           123, 113, 124, 129, 116, 104, 119, 114, 104, 120, 
                           110, 109, 107
                        ])

    kde_dark = KernelDensity(kernel='gaussian', bandwidth=2.0).fit(dark_data.reshape(-1,1))
    kde_light = KernelDensity(kernel='gaussian', bandwidth=2.0).fit(light_data.reshape(-1,1))


    def __init__(self, currState, color: str, obs:int):
        self.state = currState
        self.color: str = color #The color(light/dark) on the map at self.state
        self.weight:float = self.prob(obs, self.state)
    
    def setWeight(self, obs:int):
        self.weight = self.prob(obs)
    
    def setState(self, newState):
        self.state = newState
    
    def prob(self, obs:int):
        if(self.color == "dark"):
            return self.darkProb(obs)
        elif(self.color == "light"):
            return self.lightProb(obs)
        pass 
    
    def darkProb(self, obs:int):
        log_prob = Particle.kde_dark.score_samples([[obs]])[0]
        return np.exp(log_prob)

    def lightProb(self, obs:int):
        log_prob = Particle.kde_light.score_samples([[obs]])[0]
        return np.exp(log_prob)

