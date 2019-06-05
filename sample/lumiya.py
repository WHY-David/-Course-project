#  GNU General Public License for more details.
#dv1=5edition
#---------------------------露 米 娅--------------------------
#此代码为公开版本
#鉴于已出现大量该代码的副本，在此对该代码的主要技术细节作简要注释
#4笨蛋的代码与该代码区别仅在于参数的微调
#-----------5.20更新-----------
#吃球函数中存在修正系数0<r1<1,原吃(狙)球函数将角度缩减至r1**4，因为加速度的离散性容易在速度较小时出现
#到处乱喷球的情况，现已更新新的吃球函数stra3(sel,cel),r1采取分段函数的形式，经测定对比使用原吃球函数
#的代码胜率超过85%概率在60%-63%之间，除了避免乱喷球的状况以外，吃球也更加大胆
#本代码仍使用原吃球函数

from consts import Consts
from math import *
from random import randrange

class Player():
    def __init__(self, id, arg = None):
    	self.id = id
    	self.target=None

    

    def strategy(self, allcells):
        #基本几何函数---------------------------------
        def norm(v):#返回v的2范数
            return sqrt(v[0]**2+v[1]**2)
        
        def dianz(a,b,c):#3坐标a,b,c,返回a到直线bc的距离
            u1=[c[0]-b[0],c[1]-b[1]]
            u2=[a[0]-b[0],a[1]-b[1]]
            u=abs(u1[0]*u2[0]+u1[1]*u2[1])/norm(u1)
            a=u2[0]**2+u2[1]**2-u**2
            if a<1e-6:
                return 0
            else:
                return sqrt(a)

        def thet(a,b):#向量a到b的有向角([0,2pi))
            if sqrt((a[0]**2+a[1]**2)*(b[0]**2+b[1]**2))<1e-6:
                return 0
            det=a[0]*b[1]-a[1]*b[0]
            jia=(a[0]*b[0]+a[1]*b[1])/sqrt((a[0]**2+a[1]**2)*(b[0]**2+b[1]**2))
            if abs(jia)>1-1e-3:
                if jia<0:
                    return pi
                else:
                    return 0
            jia=acos(jia)
            if det>0:
                return 2*pi-jia
            else:
                return jia

        def chuan(v):#穿屏最小向量(例如坐标[1,1]到坐标[999,499]之间不穿屏向量是[998,498]穿屏向量则是[-2,-2])
            #只有坐标会用到穿屏函数，速度无论穿不穿都是一样的，不需要穿屏
            nonlocal WX,WY
            lst=[v[0]-WX,v[0],v[0]+WX]
            min1=abs(lst[0])
            i_min=0
            for i in range(3):
                if abs(lst[i])<min1:
                    i_min=i
                    min1=abs(lst[i])
            v0=lst[i_min]

            lst=[v[1]-WY,v[1],v[1]+WY]
            min1=abs(lst[0])
            i_min=0
            for i in range(3):
                if abs(lst[i])<min1:
                    i_min=i
                    min1=abs(lst[i])
            v1=lst[i_min]

            return [v0,v1]

        def jia(a,b):#向量a,b的无向角[0,pi)
            theta=thet(a,b)
            return pi-abs(pi-theta)

        def distance(sel,cel):#两球sel和cel的距离
            selp=sel.pos
            celp=cel.pos
            return norm(chuan([selp[0]-celp[0],selp[1]-celp[1]]))
            
                    
                    

        def dang(r1,r2,dist,v1):#相撞距离,参数为两球半径，dist是相距’向量‘，v1是相对速度,dang1函数是该函数参数为sel,cel
            #的版本
            return norm(dist)-r1-r2

        def time(r1,r2,dist,v1):#预测两球的相撞时间，同样time1函数是该函数参数为sel,cel的版本
            #参数(为避免出现math domain error)
            dist=chuan(dist)
            if norm(v1)<(norm(dist)-r1-r2)*0.01:
                return None
            
            if v1[0]*dist[0]+v1[1]*dist[1]<0:
                return None
            h=dianz(dist,[0,0],v1)
            if r1+r2<h+1e-3:
                return None
            else:
                l1=sqrt((r1+r2)**2-h**2)
                l2=norm(dist)**2-h**2
                if l2<1e-4:
                    return None
                l2=sqrt(l2)
                return (l2-l1)/norm(v1)

        def qie(sel,cel):#该函数考虑相对于目标cel相撞还需行进的距离(time1(sel,cel)*norm(v))
            r1=sel.radius
            r2=cel.radius
            dist=[cel.pos[0]-sel.pos[0],cel.pos[1]-sel.pos[1]]
            dist=chuan(dist)
            v1=[sel.veloc[0]-cel.veloc[0],sel.veloc[1]-cel.veloc[1]]

            #参数
            dist=chuan(dist)
            if norm(v1)<(norm(dist)-r1-r2)*0.01:
                return None
                    
            if v1[0]*dist[0]+v1[1]*dist[1]<0:
                return None
            h=dianz(dist,[0,0],v1)
            if r1+r2<h+1e-3:
                return None
            else:
                l1=sqrt((r1+r2)**2-h**2)
                l2=norm(dist)**2-h**2
                if l2<1e-4:
                    return None
                l2=sqrt(l2)
                return (l2-l1)
        #策略函数-----------------------------------
        def stra1(sel,cel):#吃球函数
            #sel不会朝着目标的方向加速，而是与该方向呈一定夹角的角度，该夹角的计算依赖于相对速度与该方向的夹角
            #具体详见下面代码
            #p1,p2分别为自己和目标的坐标，v1位相对速度，a为自己朝向目标的向量，其他参数则不需要使用
            nonlocal dv2
            p1=sel.pos
            p2=cel.pos
            v1=[sel.veloc[0]-cel.veloc[0],sel.veloc[1]-cel.veloc[1]]
            a=[p2[0]-p1[0],p2[1]-p1[1]]
            a=chuan(a)
            p2=[p1[0]+a[0],p1[1]+a[1]]
            p3=[p1[0]+v1[0],p1[1]+v1[1]]
            
            b=[p3[0]-p1[0],p3[1]-p1[1]]
            c=[p2[0]-p3[0],p2[1]-p3[1]]
            theta1=thet([-v1[0],-v1[1]],c)
            if theta1>pi:
                theta1=theta1-2*pi

            #r1即为所说的修正系数，下面的stra3函数修改了此处的r1
            #r1为重要参数(函数参数)*****************，不同的r1意味着不同的吃球速度和开销
            r=abs(theta1/pi)
            if r>0.9:
                r1=1-5*(1-r)
            else:
                r1=r**3
            theta=theta1*r1
            return thet([0,1],[-v1[0],-v1[1]])+theta+pi
        
        def stra3(sel,cel):#分段吃球函数
            def duan(t):#duan即位所说的分段函数,duan(r1)为修正系数
                if t<0.7:
                    return t**2
                else:
                    return t
                
            nonlocal dv2
            p1=sel.pos
            p2=cel.pos
            v1=[sel.veloc[0]-cel.veloc[0],sel.veloc[1]-cel.veloc[1]]
            a=[p2[0]-p1[0],p2[1]-p1[1]]
            a=chuan(a)
            p2=[p1[0]+a[0],p1[1]+a[1]]
            p3=[p1[0]+v1[0],p1[1]+v1[1]]
            
            b=[p3[0]-p1[0],p3[1]-p1[1]]
            c=[p2[0]-p3[0],p2[1]-p3[1]]
            theta1=thet([-v1[0],-v1[1]],c)
            #在己方速度过小时不采用该吃球函数而是直接飞向目标，因为小速度意味着必须要考虑加速度的离散性，控制不好就会乱喷
            if norm(v1)<0.3:#*************
                return thet([0,1],a)+pi
            
            if theta1>pi:
                theta1=theta1-2*pi

            #比
            r=abs(theta1/pi)
            #r1为重要参数(函数参数)******************，不同的r1意味着不同的吃球速度和开销
            r1=r
            theta=theta1*duan(r1)
            return thet([0,1],[-v1[0],-v1[1]])+theta+pi

        def stra1_J(sel,cel):#吃、狙球函数,本质上与吃球函数相差不大，随意使用
            nonlocal dv2
            p1=sel.pos
            p2=cel.pos
            v1=[sel.veloc[0]-cel.veloc[0],sel.veloc[1]-cel.veloc[1]]
            a=[p2[0]-p1[0],p2[1]-p1[1]]
            a=chuan(a)
            p2=[p1[0]+a[0],p1[1]+a[1]]
            p3=[p1[0]+v1[0],p1[1]+v1[1]]
            
            b=[p3[0]-p1[0],p3[1]-p1[1]]
            c=[p2[0]-p3[0],p2[1]-p3[1]]
            theta1=thet([-v1[0],-v1[1]],c)
            if theta1>pi:
                theta1=theta1-2*pi

            #r1为重要参数******************
            r=abs(theta1/pi)
            
            r1=r
            theta=theta1*r1**3
            return thet([0,1],[-v1[0],-v1[1]])+theta+pi
            

        def stra2(p1,p2,v1,dv2):#躲球函数，直接对着目标喷球
            v=[p2[0]-p1[0],p2[1]-p1[1]]
            v=chuan(v)
            v3=[v[0]-v1[0],v[1]-v1[1]]
            return thet([0,1],v3)
        
        def guibi(sel,cell):#躲球函数(参数为sel,cel)
            nonlocal dv2
            selp=sel.pos
            selv=sel.veloc
            celp=cell.pos
            celv=cell.veloc
            return stra2(selp,celp,[selv[0]-celv[0],selv[1]-celv[1]],dv2)
        
        def time_li_x(cel):#返回time,deltar函数列表,只看比自己小的求
            nonlocal allcells
            lst1=[]
            lst2=[]
            r1=cel.radius
            for i in range(len(allcells)):
                cel1=allcells[i]
                if cel1.dead==True or cel1==cel or cel1.radius>=cel.radius:
                    lst1.append(None)
                    lst2.append(None)
                    continue
                r2=cel1.radius
                dist=[cel1.pos[0]-cel.pos[0],cel1.pos[1]-cel.pos[1]]
                v1=[cel.veloc[0]-cel1.veloc[0],cel.veloc[1]-cel1.veloc[1]]
                t=time(r1,r2,dist,v1)
                lst1.append(t)
                if t!=None:
                    lst2.append(sqrt(r1**2+r2**2)-r1)
                else:  
                    lst2.append(None)

            return [lst1,lst2]
        #预测函数-----------------------------------------
        def time_li_x_c(cel):#返回球cel在场上所有可能吃到的球及吃到他们所需的时间，吃到他们后的位置，速度，半径
            nonlocal allcells
            lst1=[]
            lst2=[]
            lst3=[]
            r1=cel.radius
            for i in range(2,len(allcells)):
                cel1=allcells[i]
                if cel1.dead==True or cel1==cel or cel1.radius>=cel.radius:
                    lst1.append(None)
                    lst2.append(None)
                    lst3.append(None)
                    continue
                r2=cel1.radius
                dist=[cel1.pos[0]-cel.pos[0],cel1.pos[1]-cel.pos[1]]
                v1=[cel.veloc[0]-cel1.veloc[0],cel.veloc[1]-cel1.veloc[1]]
                t=time(r1,r2,dist,v1)
                lst1.append(t)
                if t!=None:
                    lst2.append(sqrt(r1**2+r2**2)-r1)
                    lst3.append(jiehe_yu(cel,cel1))
                else:  
                    lst2.append(None)
                    lst3.append(None)
                

            return [lst1,lst2,lst3]

        def time_li_all_c(cel):#和上一函数相同，但同时考虑被吃的情形
            nonlocal allcells
            lst1=[]
            lst2=[]
            lst3=[]
            r1=cel.radius
            for i in range(2,len(allcells)):
                cel1=allcells[i]
                if cel1.dead==True or cel1==cel:
                    lst1.append(None)
                    lst2.append(None)
                    lst3.append(None)
                    continue
                r2=cel1.radius
                dist=[cel1.pos[0]-cel.pos[0],cel1.pos[1]-cel.pos[1]]
                v1=[cel.veloc[0]-cel1.veloc[0],cel.veloc[1]-cel1.veloc[1]]
                t=time(r1,r2,dist,v1)
                lst1.append(t)
                if t!=None:
                    lst2.append(sqrt(r1**2+r2**2)-r1)
                    lst3.append(jiehe_yu(cel,cel1))
                else:  
                    lst2.append(None)
                    lst3.append(None)
                

            return [lst1,lst2,lst3]

        def qie_li_x(cel):#该函数已废弃不用
            nonlocal allcells
            lst1=[]
            lst2=[]
            
            for i in range(len(allcells)):
                cel1=allcells[i]
                if cel1.dead==True or cel1==cel or cel1.radius>=cel.radius:
                    lst1.append(None)
                    lst2.append(None)
                    continue
                
                t=qie(cel,cel1)
                lst1.append(t)
                r1=cel.radius
                r2=cel1.radius
                if t!=None:
                    lst2.append(sqrt(r1**2+r2**2)-r1)
                else:  
                    lst2.append(None)

            return [lst1,lst2]

        def chuan1(cel):#返回cel相对sel的穿屏坐标(即sel.pos+chuan(cel.pos-sel.pos))
            nonlocal sel
            dist=[cel.pos[0]-sel.pos[0],cel.pos[1]-sel.pos[1]]
            dist=chuan(dist)
            p=[sel.pos[0]+dist[0],sel.pos[1]+dist[1]]
            return p

        def time1(sel,cel):#
            r1=sel.radius
            r2=cel.radius
            dist=[cel.pos[0]-sel.pos[0],cel.pos[1]-sel.pos[1]]
            dist=chuan(dist)
            v1=[sel.veloc[0]-cel.veloc[0],sel.veloc[1]-cel.veloc[1]]

            return time(r1,r2,dist,v1)
        
        def jiehe(cel1,cel2):#返回两球结合后的半径和速度
            r1=cel1.radius
            r2=cel2.radius
            v1=cel1.veloc
            v2=cel1.veloc
            r=sqrt(r1**2+r2**2)
            v=[(r1**2*v1[0]+r2**2*v2[0])/(r1**2+r2**2),(r1**2*v1[1]+r2**2*v2[1])/(r1**2+r2**2)]
            return r,v

        def jiehe_yu(cel1,cel2):#返回两球结合后的位置，半径，速度
            t=time1(cel1,cel2)
            if t!=None:
                if cel1.radius>cel2.radius:
                    p1=cel1.pos
                    v1=cel1.veloc
                    p=[p1[0]+v1[0]*t,p1[1]+v1[1]*t]
                    r,v=jiehe(cel1,cel2)

                else:
                    p2=cel2.pos
                    v2=cel2.veloc
                    p=[p2[0]+v2[0]*t,p2[1]+v2[1]*t]
                    r,v=jiehe(cel1,cel2)
            return p,r,v
        
        def jiao(l1,l2):#判断两球是否已经相交
            p1=l1[0]
            p2=l2[0]
            dist=[p2[0]-p1[0],p2[1]-p1[1]]
            dist=chuan(dist)
            r1=l1[1]
            r2=l2[1]
            return norm(dist)<r1+r2

        def dang1(sel,cel):#
            r1=sel.radius
            r2=cel.radius
            dist=[cel.pos[0]-sel.pos[0],cel.pos[1]-sel.pos[1]]
            return dang(r1,r2,dist,0)


        def loss(sel,cel):#吃球损失估计，返回吃到cel需要喷球的次数，显然不能精确返回具体数值，看成是一个估计就好
            nonlocal dv2
            dist=[cel.pos[0]-sel.pos[0],cel.pos[1]-sel.pos[1]]
            dist=chuan(dist)
            v=[sel.veloc[0]-cel.veloc[0],sel.veloc[1]-cel.veloc[1]]
            theta1=(sel.radius+cel.radius)/norm(dist)
            if theta1<1:
                theta1=asin(theta1)
            jiao=jia(v,dist)
            if jiao<theta1:
                return 0
            #参数#速度越大loss越大
            else:return abs(jiao-theta1)*norm(v)/dv2*(1+norm(v)**2)+3

        def shouyi(sel,cel):#吃球收益估计，返回预测吃到该球后自身质量会变成现在质量的倍数，同上，估计就好
            nonlocal rat
            los=loss(sel,cel)
            #参数
            if (1-rat)**los<1.1*cel.radius**2/sel.radius**2:
                return None
            else:
                return (1-rat)**los/1.1+cel.radius**2/sel.radius**2

        def howl(sel,cel):#预测吃到球cel所需花费时间，估计就好
            t1=dang1(sel,cel)/max(norm(
                [sel.veloc[0]-cel.veloc[0],sel.veloc[1]-cel.veloc[1]]),0.2)
            t2=loss(sel,cel)
            return t1+t2
        
        #常数导入
        WX=Consts["WORLD_X"]
        WY=Consts["WORLD_Y"]
        dv1=Consts["DELTA_VELOC"]#喷射速度
        rat=Consts["EJECT_MASS_RATIO"]
        dv2=dv1*Consts["EJECT_MASS_RATIO"]#得到速度
        if len(allcells)<2:
            return None
        sel=allcells[self.id]
        selp=sel.pos
        selv=sel.veloc
        selr=sel.radius
        
        
        #苟--------------------------------------------
        #该游戏中苟的部分并不特别重要，不要让自己死掉的同时尽量少喷球就行，即使删掉该部分的大量代码，也不会对胜率造成太大影响
        #如果非要追求极致，可以想办法避免出现面前的小球突然吃到球变大而你没反应过来的情形，详见‘进化1‘和’进化2‘
        #找到所有比自己大的球da_li
        da_li=[]
        for x in allcells:
            if x.radius>selr and x.dead==False:
                da_li.append(x)
        #找到这些球中,碰撞所需时间最小的和碰撞所需距离最小的，他们在da_li中的下标记为i1_min和i2_min
        i2_min=None
        min2=None
        i1_min=None
        min1=None
        for i in range(len(da_li)):
            celp=da_li[i].pos
            celv=da_li[i].veloc
            t1=time1(sel,da_li[i])
            t2=qie(sel,da_li[i])

            if t2!=None:
                if i2_min==None:
                    i2_min=i
                    min2=t2
                else:
                    if t2<min2:
                        i2_min=i
                        min2=t2

            if t1!=None:
                if i1_min==None:
                    i1_min=i
                    min1=t2
                else:
                    if t1<min2:
                        i1_min=i
                        min1=t2
        #在场上存在较多大球时采用更为保守的多球方法
        if len(da_li)>5:
            if min2!=None and min2/sqrt(selr)<5:
                self.target=None
                cell=da_li[i2_min]
                celp=cell.pos
                celv=cell.veloc
                return stra2(selp,celp,[selv[0]-celv[0],selv[1]-celv[1]],dv2)
        if min1!=None and min1<60:
            self.target=None
            cell=da_li[i1_min]
            celp=cell.pos
            celv=cell.veloc
            return stra2(selp,celp,[selv[0]-celv[0],selv[1]-celv[1]],dv2)
        

        #进化算法
        d_lst=[]
        
        for i in range(len(da_li)):
            if allcells[i]!=sel:
                celp=da_li[i].pos
                celv=da_li[i].veloc
                t1=dang(selr,da_li[i].radius,chuan([celp[0]-selp[0],celp[1]-selp[1]]),[selv[0]-celv[0],selv[1]-celv[1]])
                if t1<160:
                    d_lst.append([da_li[i],t1])

        #进化1  
        for x in d_lst:
            t_l,r_l,c_l=time_li_x_c(x[0])
            for i in range(len(t_l)):
                #参数
                if t_l[i]!=None and c_l[i][1]>selr:
                    p_=[selp[0]+t_l[i]*selv[0],selp[1]+t_l[i]*selv[1]]
                    if jiao([p_,selr,selv],c_l[i])==True:
                        return guibi(sel,x[0])
                    else:
                        t=dang(selr,c_l[i][1],chuan([c_l[i][0][0]-p_[0],c_l[i][0][1]-p_[1]]),[selv[0]-c_l[i][2][0],selv[1]-c_l[i][2][1]])
                        if t!=None and t<20:
                            return guibi(sel,x[0])
                        
                        
                            
                            
                             
                            

        #进化2
        t_l,r_l,c_l=time_li_x_c(sel)
        for x in d_lst:
            for i in range(len(t_l)):
                if t_l[i]!=None:
                    if selr+r_l[i]<x[0].radius:
                        p_=[x[0].pos[0]+t_l[i]*x[0].veloc[0],x[0].pos[1]+t_l[i]*x[0].veloc[1]]
                        if jiao(c_l[i],[p_,x[0].radius])==True:
                            return guibi(sel,x[0])
                        else:
                            t=dang(c_l[i][1],x[0].radius,chuan([p_[0]-c_l[i][0][0],p_[1]-c_l[i][0][1]]),
                                   [c_l[i][2][0]-x[0].veloc[0],c_l[i][2][1]-x[0].veloc[1]])
                            if t!=None and t<12:
                                return guibi(sel,x[0])
                            
                    

                        

        

        

                        

        #吃----------------------------------------------------
        #该部分是该游戏的核心部分，半径增大的速度就是取胜的关键，同时又不能让自己的质量损失太多
        #设置target，后面有用
        target=None
        #本来有设置在大球较多时采用更为保守的方案的策略，现已废弃，故设置为100
        if len(da_li)<100:
            #选出比自己小的球
            xiao_li=[]
            for x in allcells[2:]:
                if x.radius<selr:
                    xiao_li.append(x)
            #作出吃球备选列表chi_li
            chi_li=[]#[[cel,shouyi]...]
            for cel in xiao_li:
                #首先保证距离不大
                if dang1(sel,cel)<270:#****************重要参数
                    #记录各种参数
                    los=loss(sel,cel)
                    dt=howl(sel,cel)
                    #lst用来记录该球cel在接下来的时间内可能会吃球或被吃的情况，避免出现本想吃掉小球它却突然变大的情况
                    #这种‘二次预测’的部分对胜率造成的影响不大也不小，大概在5%-15%左右，取决于参数设置
                    lst=time_li_all_c(cel)
                    #zhuang_li_r记录cel可能吃或被吃的球的半径
                    zhuang_li_r=[]
                    #首先要确定吃球的路径上不存在其他大球
                    for y in da_li:
                        a=chuan([y.pos[0]-selp[0],y.pos[1]-selp[1]])
                        b=chuan([cel.pos[0]-selp[0],cel.pos[1]-selp[1]])
                        if (dianz(chuan1(y),selp,chuan1(cel))<selr+y.radius+80 and
                            a[0]*b[0]+a[1]*b[1]>0 and a[0]*b[0]+a[1]*b[1]<norm(b)**2):
                            break
                    else:
                            
                        #然后判断该球会不会在dt时间内变为'不可吃'
                        for i in range(len(lst[0])):
                            if lst[0][i]!=None and lst[0][i]<dt:
                                zhuang_li_r.append(lst[2][i][1])
                        for r in zhuang_li_r:
                            if r>cel.radius:
                                break
                        else:
                            area1=cel.radius**2+sum([x**2 for x in zhuang_li_r])
                            if selr**2*(1-rat)**los>area1:
                                #所有情况检查完毕，将它放到chi_li列表，并附上对其收益的预测量
                                shouyi1=shouyi(sel,cel)
                                if shouyi1!=None:
                                    chi_li.append([cel,shouyi1])
            #选出最大收益球
            x_ma=None
            max1=None
            for u in chi_li:
                if x_ma==None:
                    x_ma=u[0]
                    max1=u[1]
                else:
                    if max1<u[1]:
                        max1=u[1]
                        x_ma=u[0]
                #执行吃球，此处设置max1>1.15因为损失函数和收益函数都不是精确的，保守起见在预测收益较大时才选择吃球
                if x_ma!=None and max1>1.15:
                    #所有条件满足，设置target
                    target=x_ma
                    v=[sel.veloc[0]-target.veloc[0],sel.veloc[1]-target.veloc[1]]
                    #此处的2个条件(1)吃到目标球的角度还没对，体现在loss>0,或者吃到目标球的速度还不大，体现在norm(v)<参数
                    #0.3********************重要参数
                    if norm(v)<0.3 or loss(sel,target)>0:
                        return stra1_J(sel,target)#更改吃球函数

        #狙--------------------------------------------  
        #狙击对手的部分同样不是本游戏的重点，它的重要性甚至比苟的部分还低，删掉该部分对胜率造成的影响也不大
        #这是因为，只有在优势情形下才会选择狙击对手，但是本游戏翻盘的希望不大，这是一个半径越大越容易，越小越难的游戏
        #所以，狙击的意义很多时候仅仅在于快速结束比赛
        #即便如此，该部分还是对胜率有所提升，同时使得比赛更富有观赏性
        #如何有效狙击对手同时不把自己玩死是该部分主要考虑的因素
        #设立target条件避免在正在吃球时去狙击对手，影响自己变大的效率，得不偿失
        if target==None:
            #riv用来记录对手的球,dist位置向量，v相对速度
            riv=allcells[1-self.id]
            dist=[riv.pos[0]-selp[0],riv.pos[1]-selp[1]]
            dist=chuan(dist)
            v=[selv[0]-riv.veloc[0],selv[1]-riv.veloc[1]]
            if riv.radius>5:#在对手的半径已经毫无希望的时候不再追击，因为对手半径越小，狙击越难
                #下面的一对条件判别采用了分段判别，因为连续的判别并不好做，故分段调参
                #条件中所有参数均为重要参数*********************过大会导致胜率降低
                if (riv.radius<selr*0.8 and 
                    distance(sel,riv)<200 and jia(dist,v)<pi/6 or
                    riv.radius<selr*0.65 and 
                    distance(sel,riv)<300 and jia(dist,v)<pi/3 or
                    riv.radius<selr*0.75 and norm(v)<0.3 and distance(sel,riv)<200 or
                    riv.radius<selr*0.55 and norm(v)<0.6 and distance(sel,riv)<300):
                    
                    #同样在狙击的路上不存在大球的时候才选择狙击
                    for y in da_li:
                        #参数
                        a=chuan([y.pos[0]-selp[0],y.pos[1]-selp[1]])
                        b=chuan([riv.pos[0]-selp[0],riv.pos[1]-selp[1]])
                        if (dianz(chuan1(y),selp,chuan1(riv))<selr+y.radius+80 and
                            a[0]*b[0]+a[1]*b[1]>0 and a[0]*b[0]+a[1]*b[1]<norm(b)**2):
                            break
                    else:#在场上大球数量不同时采用不同的狙击方案，大球数量<=1时追击更为狂野
                        #在己方速度不大(避免追球使自己进入高速状态，速度越大意味着危险越大)时，相对riv速度不大
                        #或者角度不对就进行追击
                        #此处选择stra1为追击函数，实际上选哪个意义不大，因为追击一般都发生在一条直线上
                        #norm(selv)后的参数为重要参数****************过大会导致胜率降低
                        if len(da_li)>1:
                            if norm(selv)<0.5 and (qie(sel,riv)==None or norm(v)<0.7):
                                return stra1(sel,riv)
                        elif len(da_li)==1:
                            if norm(selv)<1 and (qie(sel,riv)==None or norm(v)<1):
                                return stra1(sel,riv)
                            
                        else:
                            if qie(sel,riv)==None or norm(v)<1.8:
                                return stra1(sel,riv)
                            
                            
                            
#总结：本代码为基础代码，设置好参数后对该版本的胜率可以超过70%，重要的参数都已用*号标出
#参数的调试往往是非常漫长且无聊的过程，因为若要研究某个参数对胜率的影响，往往参数的微调只会导致胜率上升或下降很小，有时在5%以内
#所以为了精确测定胜率变化需要进行大量的比赛，5%的胜率变化至少需要500局的比赛才能够较为准确的测出，即便是同时运行5个kernal程序也
#需要至少40min！


        
                
                
        
