# 霍夫变换在信号本底识别中的应用
凭借肉眼很难从击中图像中分辨出粒子径迹，信号本底区分算法必须分辨出一个事例中的信号和本底，从而由信号击中重建出粒子径迹。图中蓝色线为信号击中的信号丝，红色线为本底击中的信号丝。
<div align=center><img alt="信号与本底信号丝的显示" src="https://github.com/IhepML/HoughTransform/blob/master/image/signals_backgrounds.png"></div>
<h3>特征的选择</h3>
从数据中可以得到击中的三个本地特征：与靶的距离，击中时间，在击中信号丝上的沉积能量。击中的沉积能量能够很好的分辨信号和本底，因为很多本底击中都来自质子，会留下比电子更多的能量。
本地特征加上四个邻居特征（击中信号丝的左右两个信号丝的沉积能量和击中时间），可以拟合出更好的分辨效果，因为信号击中常常和其他信号击中连在一起。
GBDT分类器经过训练样本1的本地加邻居特征的拟合之后，可以对信号和本底进行初步的分类。
经过本地和邻居特征的拟合后，可以较好地分辨信号和本底，但有少量时间、能量类似信号的孤立本底无法被剔除，由于信号击中可以构成一条或几条近似圆形的轨迹，可以建立一个图形层面的特征，来排除不在轨迹上的孤立本底。
<div align=center><img src="https://github.com/IhepML/HoughTransform/blob/master/image/after_local_neigh.png"></div>
<h3>霍夫变换</h3>
划定一个潜在轨迹中心范围将其细分为大量小分区，每个小分区代表一个可能的信号轨迹中心点。每个击中跟据确定的轨迹半径来判断哪些点可能是它的轨迹中心，每个击中的可能轨迹中心构成一个以该击中为圆心，以信号轨迹半径为半径的圆。每个击中通过这种方式对潜在轨迹中心进行投票，每个击中投票时的权重是 经过本地和邻居特征进行拟合的GBDT 对每个击中是信号的概率的预测结果，这样更像信号的击中得到更多的票数，击中投票结果如下图，环绕着击中的绿色圈代表每个击中选择的潜在轨迹中心，橘色点代表获得投票的潜在轨迹中心，点的大小代表该点获得投票的多少。
<div align=center><img src="https://github.com/IhepML/HoughTransform/blob/master/image/circle_by_signals.png"></div>
通过对潜在轨迹中心的投票，我们可以选出最可能是信号轨迹中心的点，对所有潜在轨迹中心的得票进行指数变换，从而拉大信号轨迹中心与其他点的得分差距，以这个结果作为每个潜在轨迹中心的权重，对每个击中进行再投票，其投票图形如图
<div align=center><img src="https://github.com/IhepML/HoughTransform/blob/master/image/circle_by_trackcenters.png"></div>
再次投票后，每个击中会得到一个得票分数， 这个分数代表代表了每个击中像在最可能的轨迹中心的信号轨迹上的程度。
将这个霍夫变换得分作为一个新的特征与本地与邻居特征并列起来，训练一个新的GBDT，用新的GBDT对测试样本进行训练后，信号本底的分辨结果如图,新的算法剔除掉了不在信号轨迹上的孤立本底，而信号击中几乎全部保留了下来。
<div align=center><img src="https://github.com/IhepML/HoughTransform/blob/master/image/after_houghtransform.png"></div>
<h3>效率展示</h3>
下图是使用不同特征训练预测结果的ROC曲线，曲线中展示了使用不同特征训练的GBDT预测结果的信号保留率与本底排除率的变换关系，其中绿色虚线与黄色虚线分别代表了信号保留率为99%与99.7%，使用不同特征集合训练结果如下
signals retention=99%,backgrounds rejection=79.7%,96.2%,99.1%
signals retention=99.7%,backgrounds rejection=67.5%,92.1%,98.3%
<div align=center><img src="https://github.com/IhepML/HoughTransform/blob/master/image/roc_curve.png></div>
<table align=center border="1" cellspacing="0">
    <tr>
        <td width="60">sig.ret</td>
        <td style="color:red" width="60">local</td>
        <td style="color:blue" wodth="60">loc+neigh</td>
        <td style="color:green" width="60">loc+neigh+hough</td>
    </tr>
    <tr>
        <td>99%</td>
        <td style="color:red">79.7%</td>
        <td style="color:blue">96.2%</td>
        <td style="color:green">99.1%</td>
    </tr>
    <tr>
        <td>99.7%</td>
        <td style="color:red">67.5%</td>
        <td style="color:blue">92.1%</td>
        <td style="color:green">98.3%</td>
    </tr>
</table>
