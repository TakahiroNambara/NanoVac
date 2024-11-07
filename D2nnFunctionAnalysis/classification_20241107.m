





% output region setting
    shifter=[0 0];
    offset=35;
    width=20;
% end setting

       
       temp=imresize(InputData,[120 120]);   % input image data as " A "

       ANS=padarray(temp,[round((256-size(temp,1))./2),round((256-size(temp,1))./2)],0);
  
       ans1=sum(ANS(68+offset+shifter(2):68+width+offset+shifter(2),68+offset+shifter(1):68+width+offset+shifter(1)),'all');
       ans5=sum(ANS(68+offset+shifter(2):68+width+offset+shifter(2),256-68-width-offset+shifter(1):256-68-offset+shifter(1)),'all');
       ans6=sum(ANS(256-68-width-offset+shifter(2):256-68-offset+shifter(2),68+offset+shifter(1):68+width+offset+shifter(1)),'all');
       ans7=sum(ANS(256-68-width-offset+shifter(2):256-68-offset+shifter(2),256-68-width-offset+shifter(1):256-68-offset+shifter(1)),'all');

       answer=[ans1 ans5 ans6 ans7]; % sum of inteinsity 
       answerNorm=answer./max(answer,[],'all'); % Normalized Intensity

        figure (1)
        XX=categorical({'1','5','6','7'});
        bar(XX,answerNorm)
        pbaspect([1 1 1])
        title Classification Resutls




