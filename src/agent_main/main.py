from simple_agent import SimpleAgent2

bot = SimpleAgent2(initial_roi_threshold=0.25, ending_roi_threshold=1.45)

def agent(obs):
    return bot.predict(obs)