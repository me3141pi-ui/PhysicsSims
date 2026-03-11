from heatFlow import heat2d
x = heat2d.heat2d(n = 512,init_temp=100,init_cond=100)


x.simulate_visual(1000000,0.1,cnt_src=[(200,210,200,210,273),(220,230,220,230,0)])