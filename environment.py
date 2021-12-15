import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Box:
    """
    Simple container where rectandles are stuffed in. 
    """
    
    def __init__(self, 
                 size_x : float, 
                 size_y : float) -> None:

        self.size_x = size_x
        self.size_y = size_y
        self.x = 0 
        self.y = 0 
        

class Rectangle: 
    """
    Data structures for the rectangle objects that will be stuffed inside 
    a box. 
    
    Few things to notice:    
    - Positions always refer to lower left corner. 
    
    - Rotation means 90 degree roration that is done by swapping the 
    size_x and size_y parameters. 
    """
    
    
    def __init__(self, 
                 name : str, 
                 size_x : float, 
                 size_y : float, 
                 color : list) -> None:
        
        self.name = name
        
        # Size of the rectangle
        self._size_x = size_x
        self._size_y = size_y
        
        # Position of lower left corner
        self._x = 0 
        self._y = 0 
        
        # Rotation flag 
        self.rotated = False
            
        # Visualization params
        self.color = color 

        
    @property
    def x(self) -> float: 
        return self._x


    @x.setter
    def x(self, value : float) -> None: 
        self._x = value

        
    @property
    def y(self) -> float: 
        return self._y

        
    @y.setter
    def y(self, value : float) -> None: 
        self._y = value

        
    @property
    def size_x(self) -> float: 
        if self.rotated: 
            return self._size_y
        else:
            return self._size_x

        
    @property
    def size_y(self) -> float: 
        if self.rotated: 
            return self._size_x
        else:
            return self._size_y


class Environment:
    """
    Environment for the box packing problem. 
    """
    
    def __init__(self, 
                 fill_ratio : float = 0.75):
        
        self.fill_ratio = fill_ratio
        self.box_size_x = 4
        self.box_size_y = 6
    
    
    def reset(self): 
        # Initialize seed value for reproducing the demo case 
        np.random.seed(313)

        # Create box object
        self.box = Box(size_x=self.box_size_x, size_y=self.box_size_y)
        box_area = self.box_size_x * self.box_size_y

        self.rectangles = {}
        rect_count = 0 
        rect_areas = 0
        while (rect_areas / box_area) < self.fill_ratio: 
            size_x = 0.2 + np.random.rand() * 2
            size_y = 0.2 + np.random.rand() * 2
            max_rect_area = box_area * self.fill_ratio - rect_areas
            if (size_x * size_y) > max_rect_area:
                size_y = max_rect_area / size_x
                if size_y < 0.1: 
                    break
            rect_count += 1
            rect_areas += size_x * size_y 

            # Get random RGBA color for each rectangle
            color = np.random.rand(4)
            color[3] = 0.4 # make color semi-transparent

            rname = 'Rect{}'.format(rect_count)
            rect = Rectangle(name=rname, size_x=size_x, size_y=size_y, color=color)
            # Add rectangle to the box
            self.rectangles[rname] = rect

        print('Created {} rectangle objects'.format(rect_count))
        print('Rectangles are {:0.3f} % of box area'.format((rect_areas / box_area) * 100))
        
        
    def get_rectangle_names(self):
        return list(self.rectangles.keys())
        

    def deploy_chromosome(self, chromosome): 
        for name, gene in chromosome.genes.items(): 
            self.rectangles[name].x = gene.params['x']
            self.rectangles[name].y = gene.params['y']
            self.rectangles[name].rotation = gene.params['rotation'] 
        
        
    def cost(self) -> None: 
        """
        Analyze cost of constraint violations. 
        """
        # Calculate amount of overlaps between rectangles. 
        overlap_area = 0
        for rect_a in self.rectangles.values(): 
            for rect_b in self.rectangles.values(): 
                if rect_a.name == rect_b.name: 
                    continue
                overlap_area += self.overlap_area(rect_a, rect_b)
                    
        # Calculate area of rectangles outside of the box
        out_of_box_area = 0 
        for rect in self.rectangles.values(): 
            out_of_box_area += self.out_of_box_area(rect)
        
        cost = [overlap_area, out_of_box_area]
        return cost
                
        
    def out_of_box_area(self, rect : Rectangle) -> float:
        
        """
        Out of box area is calculated as difference between rect area 
        and in-box area. Result is weighted by the distance from the box. 
        """
        rect_area = rect.size_x * rect.size_y
        in_box_area = self.overlap_area(rect, self.box)
        outside_area_cost = rect_area - in_box_area
        
        if outside_area_cost > 0: 
            rect_cp = [rect.x + rect.size_x / 2, rect.y + rect.size_y / 2]
            box_cp = [self.box.x + self.box.size_x / 2, self.box.y + self.box.size_y / 2]
            dist = np.linalg.norm(np.array(rect_cp) - np.array(box_cp))
            outside_area_cost = outside_area_cost * (dist ** 1)
        
        return outside_area_cost
        
        
    def overlap_area(self, 
                     rect_a : Rectangle, 
                     rect_b : Rectangle) -> float: 
        
        """
        Calculate overlapping area of two rectangles.
        """
        # Test for no overlap cases
        if rect_a.x + rect_a.size_x  <= rect_b.x: 
            return 0
        if rect_b.x + rect_b.size_x <= rect_a.x: 
            return 0
        if rect_a.y + rect_a.size_y  <= rect_b.y: 
            return 0
        if rect_b.y + rect_b.size_y  <= rect_a.y: 
            return 0

        # At this point the rectangles are known to overlap 
        x_overlap = min(
            rect_a.size_x, # a is smaller than b and a is totally inside b
            rect_b.size_x, # b is smaller than a and b is totally inside a
            rect_a.x + rect_a.size_x - rect_b.x, # a is at lower position than b
            rect_b.x + rect_b.size_x - rect_a.x # b is at lower position than a
        )
        y_overlap = min(
            rect_a.size_y, # a is smaller than b and a is totally inside b            
            rect_b.size_y, # b is smaller than a and b is totally inside a
            rect_a.y + rect_a.size_y - rect_b.y, # a is at lower position than b 
            rect_b.y + rect_b.size_y - rect_a.y # b is at lower position than a
        )
        return x_overlap * y_overlap
        
        
    def render(self,
               fig_size : tuple = (10, 10), 
               font_size : float = 12,
              ) -> None: 
        
        plt.rcParams.update({'font.size': font_size})
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')

        # Plot box borders as black line
        x1 = self.box.x
        x2 = self.box.x + self.box.size_x
        y1 = self.box.y
        y2 = self.box.y + self.box.size_y
        plt.plot(
            [x1, x2, x2, x1, x1],
            [y1, y1, y2, y2, y1],
            lw=2,
            color='black'
        )

        # Plot the rectangles
        for rect in self.rectangles.values():
            rect_patch = patches.Rectangle(
                xy=(rect.x, rect.y),
                width=rect.size_x,
                height=rect.size_y,
                linewidth=2,
                edgecolor='dimgrey',
                facecolor=rect.color
            )
            ax.add_patch(rect_patch)
            # Plot some dummy markers just to get pyplot plotting auto scale 
            # behaving correctly. 
            plt.scatter(rect.x, rect.y, s=1)
            plt.scatter(rect.x + rect.size_x, rect.y + rect.size_y, s=1)

            text = rect.name
            plt.annotate(
                xy=(rect.x + rect.size_x / 2, rect.y + rect.size_y / 2),
                s=text,
                ha='center',
                va='center',
                rotation=90 if (rect.size_x < rect.size_y) else 0
            )
        plt.show()
        
        
