fov = 69.4;
fov_rad = 69.4*pi/180;
bot_location = [2,1];
bot_angle = pi/2;
rot2d = @(theta) [cos(theta) -sin(theta);sin(theta) cos(theta)];
min_d = 0.5;
max_d = 6;
frustrum_t0 = bot_location+[min_d,0]*rot2d(-fov_rad/2-bot_angle);
frustrum_b0 = bot_location+[min_d,0]*rot2d(fov_rad/2-bot_angle);
frustrum_t1 = bot_location+[max_d,0]*rot2d(-fov_rad/2-bot_angle);
frustrum_b1 = bot_location+[max_d,0]*rot2d(fov_rad/2-bot_angle);
figure(1)
xlim([-15 15])
for i = 0:0.1:5
    j = 5-i;
    k = i;
    bot_location = [i,j];
    bot_angle = k;
    frustrum_t0 = bot_location+[min_d,0]*rot2d(-fov_rad/2-bot_angle);
    frustrum_b0 = bot_location+[min_d,0]*rot2d(fov_rad/2-bot_angle);
    frustrum_t1 = bot_location+[max_d,0]*rot2d(-fov_rad/2-bot_angle);
    frustrum_b1 = bot_location+[max_d,0]*rot2d(fov_rad/2-bot_angle);
    figure(1)
    clf
    hold on
    
    plot(bot_location(1),bot_location(2),'bo')
    plot(frustrum_t0(1),frustrum_t0(2),'r.')
    plot(frustrum_t1(1),frustrum_t1(2),'r.')
    plot([frustrum_t0(1),frustrum_t1(1)],[frustrum_t0(2),frustrum_t1(2)],'r')
    plot(frustrum_b0(1),frustrum_b0(2),'r.')
    plot(frustrum_b1(1),frustrum_b1(2),'r.')
    plot([frustrum_b0(1),frustrum_b1(1)],[frustrum_b0(2),frustrum_b1(2)],'r')
    plot([frustrum_t0(1),frustrum_b0(1)],[frustrum_t0(2),frustrum_b0(2)],'r')
    plot([frustrum_t1(1),frustrum_b1(1)],[frustrum_t1(2),frustrum_b1(2)],'r')
%     xlim([0 10])
%     ylim([-0.4 0.8])
    xlim([-15 15]);
    ylim([-15 15]);
end