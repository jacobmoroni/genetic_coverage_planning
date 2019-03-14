px = [41, 34, 22, 52]'
in = [98,85,69,117]'
plot(px, in, '.')
hold on
cfit = fit(px,in,'poly1')
plot(cfit)

41/93
34/85
34/105
22/69
52/117
% scale = (41/93 + 34/85 + 52/117)/3