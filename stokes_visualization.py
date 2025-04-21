from manim import *
import numpy as np
import math

class StokesSecondProblemVisualization(Scene):
    def construct(self):
        # Setting basic parameters
        self.time_elapsed = 0
        amplitude = 0.4
        frequency = 2
        omega = frequency * PI  # Angular frequency
        
        # Stokes second problem characteristic length (δ = √(2ν/ω))
        viscosity = 1.2  # Increased viscosity for longer delta
        delta = np.sqrt(2 * viscosity / omega)  # Characteristic length
        
        # Create title
        title = Text("Stokes Second Problem Visualization", font_size=36)
        subtitle = Text("Reference Frame Transformation: From Particle to Boundary Vibration", font_size=24)
        title_group = VGroup(title, subtitle).arrange(DOWN)
        title_group.to_edge(UP)
        self.play(Write(title_group))
        
        # Create boundary and particle - using left side of screen
        boundary = Line([-5, -4, 0], [-5, 4, 0], color=BLUE, stroke_width=4)
        
        # Add tick marks
        ticks = VGroup()
        for i in range(-8, 9):
            tick = Line([-5, i/2, 0], [-4.9, i/2, 0], color=BLUE)
            label = MathTex(f"{i/2}").scale(0.4).next_to(tick, LEFT, buff=0.1)
            ticks.add(VGroup(tick, label))
        
        boundary_with_ticks = VGroup(boundary, ticks)
        
        # Create particle and label
        particle = Dot([-4.2, 0, 0], color=RED).scale(1.5)
        particle_label = Text("Particle", font_size=20).next_to(particle, RIGHT, buff=0.2)
        
        # Display initial scene
        self.play(
            FadeIn(boundary_with_ticks),
            FadeIn(particle),
            Write(particle_label),
            FadeOut(particle_label)
        )
        
        # Create laser representation - from top down
        laser_cone = VGroup()
        # Laser cone top surface (source)
        laser_source = Circle(radius=0.2, color=GREEN).set_fill(GREEN, opacity=0.8)
        laser_source.move_to([-4.2, 0, 0] + UP*2)  # Above the particle
        
        # Laser beam (from source to particle)
        laser_beam = VGroup()
        beam_width = 0.15
        for i in range(8):
            opacity = 0.7 - i * 0.08
            beam = Triangle(fill_opacity=opacity, fill_color=GREEN, stroke_width=0)
            beam.scale(beam_width * (1 + i/10))
            beam.move_to(laser_source.get_center() - UP*i*0.25)
            beam.rotate(PI)  # Point triangle tip downward
            laser_beam.add(beam)
        
        laser_label = Text("Laser", font_size=16).next_to(laser_source, RIGHT, buff=0.1)
        laser_cone = VGroup(laser_source, laser_beam, laser_label)
        
        # Add laser
        self.play(
            FadeIn(laser_cone)
        )
        
        # Add vibration indication
        vibration_text = Text("Harmonic \n Oscillation", font_size=15)
        vibration_text.next_to(particle, DOWN, buff=0.3)
        self.play(Write(vibration_text))
        
        # Set up particle vibration path
        path = TracedPath(particle.get_center, stroke_width=2, stroke_color=BLUE, stroke_opacity=0.5)
        self.add(path)
        
        # Particle vibration animation
        def update_particle(mob, dt):
            self.time_elapsed += dt
            t = self.time_elapsed
            new_y = amplitude * np.sin(omega * t)
            mob.move_to([-4.2, new_y, 0])
            particle_label.next_to(mob, RIGHT, buff=0.2)
            
            # Update laser end position
            laser_beam[7].move_to(mob.get_center() + UP*0.1)
        
        particle.add_updater(update_particle)
        self.wait(4)
        
        # Stop vibration
        particle.remove_updater(update_particle)
        self.remove(path)
        
        # Transition to reference frame transformation
        self.play(
            FadeOut(title_group),
            FadeOut(vibration_text)
        )
        
        reference_text = Text("Transforming to Particle Reference Frame...", font_size=32)
        reference_text.to_edge(UP)
        self.play(Write(reference_text))
        self.wait(2)  # Display for 2 seconds
        self.play(FadeOut(reference_text))
        
        # Reference frame text - keep same reference frame
        ref_system_text = Text("Boundary Frame", font_size=20)
        ref_system_text.to_corner(UR)
        self.play(
            FadeIn(ref_system_text)
        )
        
        # Update explanation text
        final_text = Text("Particle Stationary, Boundary Vibrating", font_size=32)
        final_text.to_edge(UP)
        self.play(Write(final_text))
        self.wait(2)  # Display for 2 seconds
        self.play(FadeOut(final_text))
        
        # Reset time
        self.time_elapsed = 0
        
        # Add boundary vibration path
        boundary_path = TracedPath(
            lambda: boundary_with_ticks.get_center() + UP * 0.1,
            stroke_width=2,
            stroke_color=BLUE_E,
            stroke_opacity=0.8
        )
        self.add(boundary_path)
        
        # Create flow field visualization - IMPROVED VERSION
        flow_field_container = VGroup()
        self.add(flow_field_container)
        
        # Increase density of arrow grid
        x_positions = np.linspace(-4.9, 7, 50)  # More x positions and extended range
        y_positions = np.linspace(-3.5, 3.5, 25)  # More y positions
        
        # Create arrow object pool - for optimization
        arrows_pool = {}
        for x_pos in x_positions:
            if x_pos > boundary.get_center()[0]:  # Only on right side of boundary
                for y_pos in y_positions:
                    key = (x_pos, y_pos)
                    # Create initial arrow (length 0) - will be updated by the updater function
                    arrow = Arrow(
                        start=np.array([x_pos, y_pos, 0]),
                        end=np.array([x_pos, y_pos, 0]),
                        buff=0,
                        stroke_width=2.5,  # Increased thickness
                        color=WHITE,
                        max_stroke_width_to_length_ratio=5,
                        max_tip_length_to_length_ratio=0.35
                    )
                    arrows_pool[key] = arrow
        
        # Create envelope curve objects
        envelope_top = VMobject(stroke_color=YELLOW, stroke_width=2.5, stroke_opacity=0.9)  # Improved visibility
        envelope_bottom = VMobject(stroke_color=YELLOW, stroke_width=2.5, stroke_opacity=0.9)
        
        
        # Create derivation process box - MOVED TO RIGHT SIDE
        derivation_box = Rectangle(
            width=5.5,  # Width
            height=8.5,  # Height
            fill_color=BLACK,
            fill_opacity=0.8,
            stroke_color=WHITE
        )
        derivation_box.to_edge(RIGHT, buff=0.3)  # CHANGED FROM LEFT TO RIGHT
        
        # Add derivation process steps with improved formatting
        derivation_steps = []
        
        # Step 1: NS equation
        step1 = MathTex(r"\frac{\partial u}{\partial t} = \nu \frac{\partial^2 u}{\partial y^2}", font_size=26)
        derivation_steps.append(step1)
        
        # Step 2: Boundary condition
        step2 = MathTex(r"u = U\cos \omega t \quad \text{at } y=0", font_size=26)
        derivation_steps.append(step2)
        
        # Step 3: Other boundary condition
        step3 = MathTex(r"u \to 0 \quad \text{as } y \to \infty", font_size=26)
        derivation_steps.append(step3)
        
        # Step 4: Assumed solution form
        step4 = MathTex(r'\text{``Guessing" solution } u = f(y) e^{i\omega t} \text{ (+ c.c.)}', font_size=24)
        derivation_steps.append(step4)
        
        # Step 5: Substituting into equation
        step5 = MathTex(r"i\omega f = \nu \frac{d^2f}{dy^2}", font_size=26)
        derivation_steps.append(step5)
        
        # Step 6: Characteristic equation - split into two lines
        step6a = MathTex(r"\frac{d^2f}{dy^2} - \lambda^2 f = 0", font_size=24)
        step6b = MathTex(r"\text{where } \lambda^2 = \frac{i\omega}{\nu}", font_size=24)
        step6 = VGroup(step6a, step6b).arrange(DOWN, aligned_edge=LEFT)
        derivation_steps.append(step6)
        
        # Step 7: Solution form
        step7 = MathTex(r"f(y) = Ae^{\lambda y} + Be^{-\lambda y}", font_size=26)
        derivation_steps.append(step7)
        
        # Step 8: Characteristic value calculation - split into multiple lines
        step8a = MathTex(r"\lambda = \sqrt{\frac{i\omega}{\nu}} = \sqrt{\frac{\omega}{\nu}} e^{i\pi/4}", font_size=22)
        step8b = MathTex(r"\Rightarrow \lambda = \frac{1+i}{\sqrt{2}} \sqrt{\frac{\omega}{\nu}}", font_size=22)
        step8 = VGroup(step8a, step8b).arrange(DOWN, aligned_edge=LEFT)
        derivation_steps.append(step8)
        
        # Step 9: Simplified expression
        step9 = MathTex(r"\lambda = k(1+i) \text{ where } k = \sqrt{\frac{\omega}{2\nu}}", font_size=24)
        derivation_steps.append(step9)
        
        # Step 10: Expanded solution
        step10 = MathTex(r"f(y) = Ae^{ky}e^{iky} + Be^{-ky}e^{-iky}", font_size=26)
        derivation_steps.append(step10)
        
        # Step 11: Applying boundary conditions - split into two lines
        step11a = MathTex(r"B=U \text{ since } y \to \infty \Rightarrow f(y) \to 0", font_size=24)
        step11b = MathTex(r"\Rightarrow u(y,t) = Ue^{-ky}e^{-iky}e^{i\omega t}", font_size=24)
        step11 = VGroup(step11a, step11b).arrange(DOWN, aligned_edge=LEFT)
        derivation_steps.append(step11)
        
        # Step 12: Final solution
        step12a = MathTex(r"\text{Take real part: }", font_size=22)
        step12b = MathTex(r"u(y,t) = Ue^{-ky}\cos(\omega t - ky)", font_size=22)
        step12c = MathTex(r"k = \sqrt{\frac{\omega}{2\nu}}", font_size=22)
        step12 = VGroup(step12a, step12b, step12c).arrange(DOWN, aligned_edge=LEFT)
        derivation_steps.append(step12)
        
        # Step 13: Explanation
        step13 = Text("Damped traveling wave with \n exponential amplitude decay", font_size=18)
        derivation_steps.append(step13)
        
        # Step 14: Characteristic length - split into clearer components
        step14a = MathTex(r"\text{Characteristic Length: } \delta = \frac{1}{k} = \sqrt{\frac{2\nu}{\omega}}", font_size=22)
        step14b = MathTex(r"\text{Decay } \sim \left(\frac{1}{e}\right)^{y/\delta}", font_size=22)
        step14 = VGroup(step14a, step14b).arrange(DOWN, aligned_edge=LEFT)
        derivation_steps.append(step14)
        
        # Display derivation box
        self.play(Create(derivation_box))
        
        # MODIFIED: Changed to accumulate steps with older ones disappearing when out of space
        all_steps = VGroup()
        displayed_steps = []  # Track currently displayed steps
        max_visible_steps = 6  # Maximum visible steps at once
        
        # Calculate starting position (top of the box with some padding)
        start_pos = derivation_box.get_top() + DOWN * 0.5
        current_pos = start_pos.copy()
        
        # Display steps one by one with accumulating display
        for i, step in enumerate(derivation_steps):
            # Position the step at the current position
            step.move_to(current_pos, aligned_edge=UP)
            
            # Write the step
            self.play(Write(step), run_time=0.2)
            displayed_steps.append(step)
            
            # Add to steps and update position for next step
            all_steps.add(step)
            
            # Update position for next step
            current_pos = step.get_bottom() + DOWN * 0.2
            
            # Check if we need to remove old steps to make space
            if len(displayed_steps) > max_visible_steps:
                # Remove the oldest step
                oldest_step = displayed_steps[0]
                displayed_steps.pop(0)
                
                # Fade out oldest step
                self.play(FadeOut(oldest_step), run_time=0.5)
                all_steps.remove(oldest_step)
                
                # Shift remaining steps up
                shift_amount = UP * (step.get_height() + 0.2)
                self.play(all_steps.animate.shift(shift_amount))
                current_pos += shift_amount
            
            self.wait(0.5)  # Short pause between steps
        
        
        # Boundary vibration and flow field update function - OPTIMIZED VERSION
        def update_boundary_and_flow(mob, dt):
            self.time_elapsed += dt
            t = self.time_elapsed
            
            # Update boundary position
            y_offset = amplitude * np.sin(omega * t) - mob.get_y()
            mob.shift([0, y_offset, 0])
            
            # Current boundary position
            boundary_y = mob.get_center()[1]
            boundary_x = mob.get_center()[0]
           

            # Boundary vibration velocity
            boundary_velocity = amplitude * omega * np.cos(omega * t)
            
            # Clear old flow field
            flow_field_container.remove(*flow_field_container)
            
            # Update arrows in flow field
            active_arrows = []
            for (x_pos, y_pos), arrow in arrows_pool.items():
                if x_pos > boundary_x:  # Only on right side of boundary
                    x_dist = x_pos - boundary_x  # Distance to boundary
                    
                    # Stokes second problem solution: u(x,t) = U₀e^(-x/δ)cos(ωt - x/δ)
                    # Using cosine instead of sine to get proper phase
                    velocity = boundary_velocity * np.exp(-x_dist/delta) * np.cos(omega * t - x_dist/delta)
                    
                    # Arrow length proportional to velocity
                    arrow_length = abs(velocity) * 0.4  # Scaling factor
                    if abs(arrow_length) > 0.005:  # Lower threshold to show more arrows
                        # Update arrow - using fixed y-positions not following boundary
                        arrow.become(Arrow(
                            start=np.array([x_pos, y_pos, 0]),
                            end=np.array([x_pos, y_pos + np.sign(velocity) * arrow_length, 0]),
                            buff=0,
                            stroke_width=2.5,
                            max_stroke_width_to_length_ratio=5, 
                            max_tip_length_to_length_ratio=0.35,
                            color=interpolate_color(WHITE, BLUE, abs(velocity)/abs(boundary_velocity))
                        ))
                        active_arrows.append(arrow)
            
            # Update envelope curve points with fixed position (not following boundary)
            envelope_points_top = []
            envelope_points_bottom = []
            
            # Extended envelope lines - fixed at y=0, not following boundary
            for x in np.linspace(boundary_x, 8, 60):
                x_dist = x - boundary_x
                # Amplitude decay: e^(-x/δ)
                y_amplitude = amplitude * np.exp(-x_dist/delta) * 0.9
                envelope_points_top.append([x, y_amplitude, 0])
                envelope_points_bottom.append([x, -y_amplitude, 0])
            
            # Update envelope curves
            # 手动添加两个非常接近的起始点确保曲线从边界开始
            envelope_points_top.insert(0, [boundary_x, amplitude * 0.9, 0])
            envelope_points_bottom.insert(0, [boundary_x, -amplitude * 0.9, 0])     
            
            # Add active arrows and envelope curves to container
            flow_field_container.add(*active_arrows, envelope_top, envelope_bottom)
        
        # Add updater
        boundary_with_ticks.add_updater(update_boundary_and_flow)
        
        # Add explanation box - MOVED TO LEFT SIDE
        physics_box = Rectangle(
            height=2.5,
            width=4.0,
            fill_color=BLACK,
            fill_opacity=0.8,
            stroke_color=WHITE
        )
        physics_box.to_corner(DL)  # CHANGED FROM DR TO DL
        
        # Text and MathTex combination
        physics_title = Text("Stokes Second Problem:", font_size=20)
        #physics_point1 = Text("• Boundary performs harmonic oscillation", font_size=16)
        physics_point2 = Text("• Fluid velocity follows e^(-x/δ)cos(ωt-x/δ)", font_size=16)
        physics_point3 = Text("• Characteristic length: δ = √(2ν/ω)", font_size=16)
        #physics_point4 = Text("• Phase lag increases with distance", font_size=16)
        
        physics_text = VGroup(
            physics_title,
           # physics_point1,
            physics_point2,
            physics_point3,
            #physics_point4
        ).arrange(DOWN, aligned_edge=LEFT)
        
        physics_text.move_to(physics_box.get_center())
        physics_group = VGroup(physics_box, physics_text)
        
        self.play(FadeIn(physics_group))
        self.wait(2)  # Display for 2 seconds
    #   self.play(FadeOut(physics_group))
        
        # Continue animation for long enough
        self.wait(10)
        
        # End animation
        boundary_with_ticks.remove_updater(update_boundary_and_flow)
        self.remove(boundary_path)
        
   