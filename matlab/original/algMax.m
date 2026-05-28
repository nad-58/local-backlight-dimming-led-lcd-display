function [ r ] = algMax( display, target )
%ALGMAX Performs the maximum backlight dimming algorithm
%   Description.

    r = zeros(1, display.backlightSegmentsNumber );
    for i = 1:display.backlightSegmentsNumber % for each backlight segment...
        % determine range
        d = floor([display.verticalLEDdistance display.horizontalLEDdistance] * 0.7); 
        % determine boundaries
        l = max([display.yLEDpositions(i) display.xLEDpositions(i)] - d, [1 1]);
        h = min([display.yLEDpositions(i) display.xLEDpositions(i)] + d, [display.screenHeight display.screenWidth]);
        % get image sector
        p = target(l(1):h(1), l(2):h(2), :);
        % get max value for the sector
        r(i) = max(p(:));
    end
end
