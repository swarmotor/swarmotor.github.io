
**一、选择题（每题3分，共30分）**

1. 在代码中，`#define screenx 1280` 的作用是什么？（  ）
A. 声明一个变量`screenx`并初始化为1280
B. 定义一个宏，将`screenx`替换为1280
C. 定义一个函数，返回值为1280
D. 以上都不对

2. 以下哪个结构体用于表示游戏中的豆子？（  ）
A. `BOSS`
B. `PACMAN`
C. `BEAN`
D. `WALL`

3. `initgraph(640, 480)` 函数的作用是什么？（  ）
A. 初始化一个640x480像素的图形窗口
B. 设置图形的颜色模式为640x480
C. 绘制一个640x480像素的矩形
D. 以上都不对

4. 以下哪个函数用于判断角色的移动是否合法？（  ）
A. `judgeesc`
B. `judgestep`
C. `judgestep_boss`
D. `runboss`

5. 在`rungame`函数中，`kbhit`函数的作用是什么？（  ）
A. 检测是否有键盘按键被按下
B. 获取按下的键盘按键值
C. 清除键盘缓冲区
D. 判断键盘是否可用

6. 代码中`SCORE += 150` 的作用是什么？（  ）
A. 将`SCORE`的值增加150
B. 将`SCORE`的值设置为150
C. 判断`SCORE`是否大于等于150
D. 以上都不对

7. 以下哪个函数用于从文件中读取地图数据？（  ）
A. `readmap`
B. `buildmapfile`
C. `Producenew_map`
D. `drawmap`

8. 在`drawpacman`函数中，`setfillcolor(RGB(255, 0, 0))` 的作用是什么？（  ）
A. 设置绘制线条的颜色为红色
B. 设置填充圆形的颜色为红色
C. 设置文字的颜色为红色
D. 以上都不对

9. `High_score`函数主要实现了什么功能？（  ）
A. 开始新游戏
B. 创建新地图
C. 显示高分榜
D. 退出游戏

10. 以下哪种数据结构用于存储高分记录？（  ）
A. 数组
B. 结构体
C. 链表
D. 以上都不对

**二、填空题（每题3分，共30分）**

1. 代码中`typedef struct {... } BOSS;` 的作用是____________________。
2. `GetTickCount`函数返回的是____________________。
3. 在`Producenew_map`函数中，`srand((unsigned)time(NULL));` 的作用是____________________。
4. `bean_number`变量用于记录____________________。
5. `judgestep_boss`函数中，判断敌人移动方向是否合法时，除了边界判断和与墙壁碰撞判断，还需要判断____________________。
6. `drawbean`函数通过调用`fillcircle`函数绘制豆子，`fillcircle`函数的参数包括____________________。（至少写出两个主要参数）
7. 在`writescore`函数中，`ctime(&t)` 的作用是____________________。
8. `readscore`函数从文件中读取得分记录后，通过____________________算法对得分进行排序。
9. `overwindow`函数根据传入的参数`flag`的值来判断显示胜利还是失败界面，当`flag`大于0时显示____________________。
10. 在`mainmenu`函数中，如果用户输入的选项不在1到4之间，会提示用户是否结束程序，这是通过____________________函数实现的。

**三、简答题（每题10分，共20分）**

1. 请简要描述`rungame`函数的主要功能和执行流程。
2. 解释`build`函数中构建地图中豆子布局的算法思路，包括如何确保豆子不位于地图边缘且不相邻。

**四、程序分析题（每题20分，共20分）**

1. 分析以下代码片段的功能：
```cpp
void cleanpacman(void)
{
    setcolor(RGB(0, 0, 0));
    setfillcolor(RGB(0, 0, 0));
    fillcircle(ORGX + pacman_a.x * GWID + GWID / 2, ORGY + pacman_a.y * GHEI + GHEI / 2, ROUND);
}
```
（1）这段代码实现了什么功能？
（2）如果`ROUND`的值为10，`pacman_a.x`为5，`pacman_a.y`为3，`GWID`为20，`GHEI`为20，`ORGX`为10，`ORGY`为10，计算`fillcircle`函数的圆心坐标。
（3）解释为什么在清除吃豆人绘制时需要先设置颜色为黑色。

2. 分析以下代码片段的功能：
```cpp
void runboss(void)
{
    int t1, t2;
    srand((unsigned)time(NULL));
    t1 = rand() % LEVEL;
    t2 = rand() % LEVEL;
    if (judgestep_boss(0, t1))
    {
        cleanboss_a();
        if (t1 == 1) boss_a.y--;
        if (t1 == 2) boss_a.x--;
        if (t1 == 3) boss_a.y++;
        if (t1 == 4) boss_a.x++;
    }
    if (judgestep_boss(1, t2))
    {
        cleanboss_b();
        if (t2 == 1) boss_b.y--;
        if (t2 == 2) boss_b.x--;
        if (t2 == 3) boss_b.y++;
        if (t2 == 4) boss_b.x++;
    }
    judgeesc();
    drawboss();
}
```
（1）这段代码实现了什么功能？
（2）`rand() % LEVEL` 的作用是什么？
（3）解释为什么在移动敌人后需要调用`judgeesc`和`drawboss`函数。

