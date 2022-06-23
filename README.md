# 模型在做什么
表面上，
+ 输入：编码（棋局，玩家状态）
+ 输出：a 状态对应行为的概率，b 状态的分数

哪里发挥作用了？
+ 在MCTS过程中，a 用于expand 生成概率，b 用于生成未达到节点backup时rollout的局面分数
+ 自我对弈时，根据节点生成的行为概率a',和下完一整盘的分数b'，用于和a和b的监督信号。

# MCTS在做什么

+ select：
Q' = W'/N' 
   = (W + dW)/(N+1) 
   = (W + dW)/(N + 1) - W/N + W/N 
   = (NdW - W) / (N + 1)*N + W/N 
   =  N(dW-Q)/(N+1)*N + Q
   = Q + (dW-Q)/(N+1) 
Q  = W/N
+ expand：
   
+ backup：
  

