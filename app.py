# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 12:32:10 2026

@author: d/dt Lucas
"""


import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog, ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import copy
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.cm as cm
import matplotlib.colors as colors


class FiberBuilder:

    def __init__(self, root):
        self.root = root
        self.root.title("MCF Builder")

        self.history = []

        self.nodes = []
        self.edges = []

        self.mode = "core"

        self.selected_core = None
        
        self.dragging_core = None        
        self.drag_start_saved = False

        # ── Design mode: "topology" (manual k) or "geometry" (auto-calc k) ──
        self.design_mode = "topology"   # "topology" | "geometry"

        self.geometry_threshold = 0.05
        self.geometry_neighbours = 6

        # Per-core geometry params (parallel list to self.nodes)
        # Each entry: {"radius": float [µm], "n_core": float}
        self.core_params = []

        # Global geometry parameters
        self.n_cladding = 1.444          # cladding refractive index
        self.wavelength  = 1.55e-6       # [m]  operating wavelength

        # Default values for new cores added in geometry mode
        self.default_radius  = 4.5       # [µm]
        self.default_n_core  = 1.450

        self.fig = Figure(figsize=(7,7))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)

        self.xlim = (-1.2, 1.2)
        self.ylim = (-1.2, 1.2)

        # ── Top toolbar: analysis tools ───────────────────────────────────────
        top_toolbar = tk.Frame(root)
        top_toolbar.pack(fill="x")

        tk.Button(top_toolbar, text="Show Matrix",
                  command=self.show_matrix).pack(side="left")

        tk.Button(top_toolbar, text="Show Supermodes",
                  command=self.show_supermodes).pack(side="left")

        tk.Button(
            top_toolbar,
            text="Browse Modes",
            command=self.browse_modes
        ).pack(side="left")

        tk.Button(
            top_toolbar,
            text="Propagation",
            command=self.propagation_simulator
        ).pack(side="left")

        tk.Button(
            top_toolbar,
            text="Animate Propagation",
            command=self.animate_propagation
        ).pack(side="left")

        # ── Design-mode switcher ──────────────────────────────────────────────
        tk.Label(top_toolbar, text="  |  Mode:").pack(side="left")

        self.design_mode_var = tk.StringVar(value="topology")

        ttk.Combobox(
            top_toolbar,
            textvariable=self.design_mode_var,
            values=["topology", "geometry"],
            state="readonly",
            width=10
        ).pack(side="left", padx=4)

        self.design_mode_var.trace_add(
            "write",
            lambda *_: self._on_design_mode_change()
        )

        tk.Button(
            top_toolbar,
            text="Geometry Params",
            command=self.edit_geometry_params
        ).pack(side="left", padx=4)


        # ── Canvas (fills remaining space) ────────────────────────────────────
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # ── Bottom toolbar: building tools ────────────────────────────────────
        bot_toolbar = tk.Frame(root)
        bot_toolbar.pack(fill="x")
        
        tk.Button(
            bot_toolbar,
            text="Generate Lattice",
            command=self.generate_lattice
        ).pack(side="left")

        tk.Button(bot_toolbar, text="Add Core Mode",
                  command=lambda: self.set_mode("core")).pack(side="left")

        tk.Button(bot_toolbar, text="Add Coupling Mode",
                  command=lambda: self.set_mode("coupling")).pack(side="left")

        tk.Button(
            bot_toolbar,
            text="Delete Mode",
            command=lambda: self.set_mode("delete")
        ).pack(side="left")


        tk.Button(bot_toolbar, text="Undo", command=self.undo).pack(side="left")
        
        tk.Button(bot_toolbar, text="Clear",
                  command=self.clear).pack(side="left")

        tk.Button(
            bot_toolbar,
            text="Export Figure",
            command=self.export_figure
        ).pack(side="left")



        self.canvas.mpl_connect(
            "button_press_event",
            self.on_mouse_press
        )
        
        self.canvas.mpl_connect(
            "motion_notify_event",
            self.on_mouse_move
        )
        
        self.canvas.mpl_connect(
            "button_release_event",
            self.on_mouse_release
        )
        
        

        self.redraw()
        self.root.bind("<Control-Z>", self.undo_event)
        self.root.bind("<Control-z>", self.undo_event)
        self.root.bind("1", lambda e: self.set_mode("core"))
        self.root.bind("2", lambda e: self.set_mode("coupling"))
        self.root.bind("3", lambda e: self.set_mode("delete"))
        self.root.bind("c", lambda e: self.clear())
              
    def undo_event(self, event):
        self.undo()

    def set_mode(self, mode):
        self.mode = mode
        self.selected_core = None
        self.ax.set_title(
            f"Mode: {self.mode}"
        )
        self.redraw()

    # ── Design-mode helpers ───────────────────────────────────────────────────

    def _on_design_mode_change(self):
        self.design_mode = self.design_mode_var.get()
        self.redraw()

    def _ensure_core_params(self):
        """Keep self.core_params in sync with self.nodes length."""
        while len(self.core_params) < len(self.nodes):
            self.core_params.append({
                "radius": self.default_radius,
                "n_core": self.default_n_core
            })
        if len(self.core_params) > len(self.nodes):
            self.core_params = self.core_params[:len(self.nodes)]

    def _mode_field_radius(self, a_um, n_core, n_clad):
        """
        Approximation for LP01 mode field radius w [µm].
        a_um: core radius in µm
        Returns w in µm.
        """
        lam_um = self.wavelength * 1e6          # wavelength in µm
        NA = np.sqrt(max(n_core**2 - n_clad**2, 1e-12))
        V  = (2 * np.pi * a_um / lam_um) * NA
        V  = max(V, 0.5)                        # avoid degenerate V
        # Marcuse 1978 approximation (valid for 1.2 < V < 2.4, reasonable beyond)
        w  = a_um * (0.65 + 1.619 / V**1.5 + 2.879 / V**6)
        return w

    def _calc_kappa(self, i, j):
        """
        Coupling coefficient between cores i and j [1/m  → normalised units].

        Uses the Gaussian overlap approximation with Bessel correction factor:
            κ ≈ (π·Δn·n_clad / λ) · (2·a_i·a_j / (a_i²+a_j²)) · exp(-d²/(wi²+wj²))

        For identical cores this collapses to the familiar Gaussian decay.
        Returns a dimensionless coupling scaled so that κ=1 for cores
        touching at d = a_i + a_j with V ≈ 2.
        """
        self._ensure_core_params()
        pi = self.core_params[i]
        pj = self.core_params[j]

        ai  = pi["radius"]          # µm
        aj  = pj["radius"]
        nci = pi["n_core"]
        ncj = pj["n_core"]
        ncl = self.n_cladding
        lam = self.wavelength * 1e6  # µm

        wi = self._mode_field_radius(ai, nci, ncl)
        wj = self._mode_field_radius(aj, ncj, ncl)

        # Physical distance between core centres (canvas units = µm in geom mode)
        xi, yi = self.nodes[i]
        xj, yj = self.nodes[j]
        d = np.hypot(xi - xj, yi - yj)     # µm

        # Average delta-n
        delta_n = 0.5 * ((nci - ncl) + (ncj - ncl))
        n_avg   = 0.5 * (nci + ncj)

        # Prefactor  (same structure as κ₀ in coupled-mode theory)
        k0      = 2 * np.pi / lam            # 1/µm
        prefac  = (k0 * delta_n / n_avg)     # 1/µm  (weak-guidance)

        # Size-mismatch factor (overlap of two Gaussians with different widths)
        size_fac = np.sqrt(2 * wi * wj / (wi**2 + wj**2))

        # Gaussian overlap with combined beam radius
        w_eff = np.sqrt((wi**2 + wj**2) / 2)
        gauss  = np.exp(-(d / (2 * w_eff))**2)

        kappa = prefac * size_fac * gauss   # 1/µm

        # Convert to the same "coupling units" the topology mode uses.
        # We normalise by a reference scale: coupling at d = a_ref, V_ref = 2.
        # This keeps κ ~ O(1) for typical touching-core designs.
        a_ref   = 4.5                           # µm
        n_ref   = 1.450
        w_ref   = self._mode_field_radius(a_ref, n_ref, ncl)
        dn_ref  = n_ref - ncl
        k_ref   = (k0 * dn_ref / n_ref) * np.exp(-(a_ref / w_ref)**2)
        k_ref   = max(k_ref, 1e-30)

        return float(kappa / k_ref)

    def edit_geometry_params(self):
        """
        Open a dialog to edit global geometry parameters (n_cladding, wavelength)
        and the default values for new cores.
        """
        win = tk.Toplevel(self.root)
        win.title("Geometry Parameters")
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()
        win.focus_force()

        def labeled_entry(parent, label, value, row):
            tk.Label(parent, text=label).grid(
                row=row, column=0, padx=10, pady=5, sticky="w"
            )
            var = tk.StringVar(value=str(value))
            tk.Entry(parent, textvariable=var, width=14).grid(
                row=row, column=1, padx=10, pady=5
            )
            return var

        ncl_var  = labeled_entry(win, "n_cladding:",          self.n_cladding,   0)
        lam_var  = labeled_entry(win, "Wavelength (µm):",     self.wavelength*1e6, 1)
        r_var    = labeled_entry(win, "Default radius (µm):", self.default_radius, 2)
        nc_var   = labeled_entry(win, "Default n_core:",      self.default_n_core, 3)

        def apply():
            try:
                self.n_cladding     = float(ncl_var.get())
                self.wavelength     = float(lam_var.get()) * 1e-6
                self.default_radius = float(r_var.get())
                self.default_n_core = float(nc_var.get())
            except ValueError:
                messagebox.showerror("Error", "All fields must be numbers.")
                return
            win.destroy()
            self.redraw()

        tk.Button(win, text="Apply", command=apply).grid(
            row=4, column=0, columnspan=2, pady=10
        )

    def edit_core_geometry(self, idx):
        """
        Open a dialog to edit the radius and n_core of a single core.
        """
        self._ensure_core_params()
        p = self.core_params[idx]

        win = tk.Toplevel(self.root)
        win.title(f"Core {idx} Parameters")
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()
        win.focus_force()

        tk.Label(win, text=f"Core {idx}  (x={self.nodes[idx][0]:.1f} µm, "
                           f"y={self.nodes[idx][1]:.1f} µm)",
                 font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=0, columnspan=2, padx=10, pady=6
        )

        tk.Label(win, text="Radius (µm):").grid(row=1, column=0, padx=10, pady=4, sticky="w")
        r_var = tk.StringVar(value=str(p["radius"]))
        tk.Entry(win, textvariable=r_var, width=12).grid(row=1, column=1, padx=10, pady=4)

        tk.Label(win, text="n_core:").grid(row=2, column=0, padx=10, pady=4, sticky="w")
        n_var = tk.StringVar(value=str(p["n_core"]))
        tk.Entry(win, textvariable=n_var, width=12).grid(row=2, column=1, padx=10, pady=4)

        # Computed info
        def update_info(*_):
            try:
                a  = float(r_var.get())
                nc = float(n_var.get())
            except ValueError:
                return
            lam_um = self.wavelength * 1e6
            NA     = np.sqrt(max(nc**2 - self.n_cladding**2, 1e-12))
            V      = (2 * np.pi * a / lam_um) * NA
            w      = self._mode_field_radius(a, nc, self.n_cladding)
            info_lbl.config(
                text=f"V = {V:.3f}   MFD = {2*w:.2f} µm"
            )

        info_lbl = tk.Label(win, text="", foreground="gray")
        info_lbl.grid(row=3, column=0, columnspan=2, pady=2)

        r_var.trace_add("write", update_info)
        n_var.trace_add("write", update_info)
        update_info()

        def apply():
            try:
                self.core_params[idx]["radius"] = float(r_var.get())
                self.core_params[idx]["n_core"] = float(n_var.get())
            except ValueError:
                messagebox.showerror("Error", "Fields must be numbers.")
                return
            win.destroy()
            self.redraw()

        tk.Button(win, text="Apply", command=apply).grid(
            row=4, column=0, columnspan=2, pady=10
        )

    def clear(self):
        self.nodes = []
        self.edges = []
        self.core_params = []
        self.selected_core = None
        self.redraw()
        
    def save_state(self):
        """Save a snapshot before modification."""
        self.history.append(
            (
                copy.deepcopy(self.nodes),
                copy.deepcopy(self.edges),
                copy.deepcopy(self.core_params),
                self.selected_core
            )
        )

    def undo(self):
        """Restore previous state."""
        if not self.history:
            return
    
        self.nodes, self.edges, self.core_params, self.selected_core = self.history.pop()
        self.redraw()
        
    def point_segment_distance(self, px, py, x1, y1, x2, y2):

        dx = x2 - x1
        dy = y2 - y1
    
        if dx == 0 and dy == 0:
            return np.hypot(px - x1, py - y1)
    
        t = (
            ((px - x1) * dx + (py - y1) * dy)
            /
            (dx*dx + dy*dy)
        )
    
        t = np.clip(t, 0, 1)
    
        projx = x1 + t*dx
        projy = y1 + t*dy
    
        return np.hypot(px - projx, py - projy)

    def nearest_core(self, x, y):

        if not self.nodes:
            return None

        dmin = 1e9
        idx = None

        for i, (cx, cy) in enumerate(self.nodes):

            d = np.hypot(cx - x, cy - y)

            if d < dmin:
                dmin = d
                idx = i

        # Use a scale-aware tolerance: 5% of the visible axis span,
        # so it works in both topology (~1 unit) and geometry (~µm) modes.
        span_x = abs(self.xlim[1] - self.xlim[0])
        span_y = abs(self.ylim[1] - self.ylim[0])
        tol = 0.05 * max(span_x, span_y, 1e-6)

        if dmin < tol:
            return idx

        return None
    
    def nearest_edge(self, x, y, tol=None):

        if tol is None:
            span_x = abs(self.xlim[1] - self.xlim[0])
            span_y = abs(self.ylim[1] - self.ylim[0])
            tol = 0.03 * max(span_x, span_y, 1e-6)

        best = None
        best_dist = 1e9
    
        for idx, (i, j, k) in enumerate(self.edges):
    
            x1, y1 = self.nodes[i]
            x2, y2 = self.nodes[j]
    
            d = self.point_segment_distance(
                x, y,
                x1, y1,
                x2, y2
            )
    
            if d < best_dist:
                best_dist = d
                best = idx
    
        if best_dist < tol:
            return best
    
        return None

    def _update_view_limits(self):
        """
        Recompute self.xlim / self.ylim from the current node positions.
        In geometry mode the margin is based on core radii; in topology mode
        a fixed fractional margin is used.
        """
        if not self.nodes:
            self.xlim = (-1, 1)
            self.ylim = (-1, 1)
            return

        xs = [p[0] for p in self.nodes]
        ys = [p[1] for p in self.nodes]

        if self.design_mode == "geometry":
            self._ensure_core_params()
            max_r = max(p["radius"] for p in self.core_params) if self.core_params else 5.0
            pad = max(max_r * 4, 10.0)          # generous padding in µm
        else:
            span = max(max(xs) - min(xs), max(ys) - min(ys), 0.1)
            pad  = 0.15 * span + 0.1             # 15 % + fixed minimum

        self.xlim = (min(xs) - pad, max(xs) + pad)
        self.ylim = (min(ys) - pad, max(ys) + pad)

    def on_mouse_press(self, event):

        if event.inaxes is None:
            return
    
        x = event.xdata
        y = event.ydata
    
        #
        # DOUBLE CLICK
        #
        if event.dblclick:

            idx = self.nearest_core(x, y)

            # In geometry mode: double-click a core → edit its params
            if idx is not None and self.design_mode == "geometry":
                self._ensure_core_params()
                self.edit_core_geometry(idx)
                return

            # In topology mode: double-click an edge → edit coupling
            if self.design_mode == "topology":
                edge_idx = self.nearest_edge(x, y)

                if edge_idx is not None:

                    i, j, old_k = self.edges[edge_idx]

                    new_k = simpledialog.askfloat(
                        "Edit Coupling",
                        "Coupling coefficient:",
                        initialvalue=old_k
                    )

                    if new_k is not None:

                        self.save_state()

                        self.edges[edge_idx] = (
                            i,
                            j,
                            new_k
                        )

                        self.redraw()

            return
    
        #
        # DELETE MODE
        #
        if self.mode == "delete":
    
            idx = self.nearest_core(x, y)
    
            if idx is not None:
    
                self.save_state()
    
                self.nodes.pop(idx)
                if idx < len(self.core_params):
                    self.core_params.pop(idx)
    
                new_edges = []
    
                for i, j, k in self.edges:
    
                    if i == idx or j == idx:
                        continue
    
                    if i > idx:
                        i -= 1
    
                    if j > idx:
                        j -= 1
    
                    new_edges.append((i, j, k))
    
                self.edges = new_edges
    
                self.redraw()
    
                return
    
            edge_idx = self.nearest_edge(x, y)
    
            if edge_idx is not None:
    
                self.save_state()
    
                self.edges.pop(edge_idx)
    
                self.redraw()
    
            return
    
        #
        # CORE MODE
        #
        if self.mode == "core":
    
            idx = self.nearest_core(x, y)
    
            if idx is not None:
    
                self.save_state()
    
                self.dragging_core = idx
    
                return
    
            self.save_state()
    
            self.nodes.append((x, y))
            # Add default geometry params for the new core
            self.core_params.append({
                "radius": self.default_radius,
                "n_core": self.default_n_core
            })
    
            if self.design_mode == "geometry":
                self._update_view_limits()
    
            self.redraw()
    
            return
    
        #
        # COUPLING MODE  (topology only)
        #
        if self.mode == "coupling":

            if self.design_mode == "geometry":
                messagebox.showinfo(
                    "Geometry Mode",
                    "Couplings are calculated automatically in Geometry mode.\n"
                    "Switch to Topology mode to set them manually."
                )
                return

            idx = self.nearest_core(x, y)
    
            if idx is None:
                return
    
            if self.selected_core is None:
    
                self.selected_core = idx
    
                return
    
            k = simpledialog.askfloat(
                "Coupling",
                "Coupling coefficient:"
            )
    
            if k is not None:
    
                existing = None
    
                for n, (i, j, oldk) in enumerate(self.edges):
    
                    if {i, j} == {self.selected_core, idx}:
                        existing = n
                        break
    
                self.save_state()
    
                if existing is None:
    
                    self.edges.append(
                        (
                            self.selected_core,
                            idx,
                            k
                        )
                    )
    
                else:
    
                    self.edges[existing] = (
                        self.selected_core,
                        idx,
                        k
                    )
    
            self.selected_core = None
    
            self.redraw()
            
    def on_mouse_move(self, event):

        if self.dragging_core is None:
            return
    
        if event.inaxes is None:
            return
    
        self.nodes[self.dragging_core] = (
            event.xdata,
            event.ydata
        )
    
        self.redraw()
        
    def on_mouse_release(self, event):

        self.dragging_core = None
                
    def on_double_click(self, event):

        if event.inaxes is None:
            return
    
        if not event.dblclick:
            return
    
        edge_idx = self.nearest_edge(
            event.xdata,
            event.ydata
        )
    
        if edge_idx is None:
            return
    
        i, j, old_k = self.edges[edge_idx]
    
        new_k = simpledialog.askfloat(
            "Edit Coupling",
            "Coupling coefficient:",
            initialvalue=old_k
        )
    
        if new_k is None:
            return
    
        self.save_state()
    
        self.edges[edge_idx] = (
            i,
            j,
            new_k
        )
    
        self.redraw()

    def coupling_matrix(self):

        N = len(self.nodes)
        C = np.zeros((N, N))

        if self.design_mode == "topology":
            # Manual topology mode: use explicit edges
            for i, j, k in self.edges:
                C[i, j] = k
                C[j, i] = k

        else:
            # Geometry mode: auto-calculate all pairwise couplings
            self._ensure_core_params()
            for i in range(N):
                for j in range(i + 1, N):
                    k = self._calc_kappa(i, j)
                    C[i, j] = k
                    C[j, i] = k

        return C

    def _edge_pairs(self, relative_threshold=0.05, max_neighbours=6):
        """
        Return a list of (i, j, k) tuples for drawing in visualisation windows.
        In topology mode: uses self.edges directly.
        In geometry mode: derives pairs from coupling_matrix, filtering weak
        couplings below `threshold` (relative to max) to avoid clutter.
        """
        if self.design_mode == "topology":
            return list(self.edges)

        N = len(self.nodes)
        if N < 2:
            return []
        C = self.coupling_matrix()
        max_k = max((abs(C[i, j]) for i in range(N) for j in range(i+1, N)), default=1e-30)
        pairs = set()

        for i in range(N):
            candidates = [(j, abs(C[i,j])) for j in range(N) if j != i]
            candidates.sort(key=lambda x: x[1], reverse=True)

            if max_neighbours is not None:
                candidates = candidates[:max_neighbours]

            for j, kij in candidates:
                if kij < relative_threshold * max_k:
                    continue
                a, b = sorted((i, j))
                pairs.add((a, b, C[a, b]))

        return list(pairs)

    def show_matrix(self):

        C = self.coupling_matrix()

        win = tk.Toplevel(self.root)
        win.title("Coupling Matrix")

        txt = tk.Text(win, width=80, height=30)
        txt.pack(fill="both", expand=True)

        txt.insert("end", str(C))

    def show_supermodes(self):
    
        C = self.coupling_matrix()
    
        if len(C) == 0:
            return
    
        eigvals, eigvecs = np.linalg.eigh(C)
    
        idx = np.argsort(eigvals)
    
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        edge_pairs = self._edge_pairs(self.geometry_threshold, self.geometry_neighbours)
        
        degeneracies = []
        tol = 1e-6        
        for i in range(len(eigvals)-1):        
            if abs(eigvals[i+1] - eigvals[i]) < tol:        
                degeneracies.append((i, i+1))
            
        N_modes = len(eigvals)
    
        ncols = int(np.ceil(np.sqrt(N_modes)))
        nrows = int(np.ceil(N_modes / ncols))
    
        #
        # Keep figure size reasonable
        #
        cell_size = 2.2
    
        fig_w = min(14, cell_size * ncols)
        fig_h = min(10, cell_size * nrows)
    
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(fig_w, fig_h),
            constrained_layout=True
        )
    
        axes = np.atleast_1d(axes).flatten()
    
        xs = [p[0] for p in self.nodes]
        ys = [p[1] for p in self.nodes]
    
        N_cores = len(self.nodes)
    
        #
        # Adaptive styling
        #
        node_scale = np.clip(
            30 / np.sqrt(max(N_cores, 1)),
            2,
            10
        )
    
        text_size = np.clip(
            14 - 0.15 * N_cores,
            4,
            10
        )
    
        edge_width = np.clip(
            3 - 0.03 * N_cores,
            0.5,
            2
        )
    
        title_size = np.clip(
            11 - 0.05 * N_cores,
            7,
            10
        )
    
        last_scatter = None
    
        for mode in range(N_modes):
    
            ax = axes[mode]
    
            field = eigvecs[:, mode]
    
            amp = np.abs(field)
    
            if amp.max() > 0:
                amp /= amp.max()
    
            field = np.real(field)
    
            if np.max(np.abs(field)) > 0:
                field /= np.max(np.abs(field))
    
            tol = 1e-8
    
            sign = np.zeros_like(field)
    
            sign[field > tol] = -1
            sign[field < -tol] = 1
    
            last_scatter = ax.scatter(
                xs,
                ys,
                s=node_scale * (100 * amp + 20),
                c=sign,
                cmap="bwr",
                vmin=-1,
                vmax=1,
                edgecolors="black",
                linewidths=0.5
            )
    
            #
            # Draw couplings
            #
            for i, j, k in edge_pairs:
    
                x1, y1 = self.nodes[i]
                x2, y2 = self.nodes[j]
    
                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color="lightgray",
                    linewidth=edge_width,
                    zorder=0
                )
    
            #
            # Only label cores if not too many
            #
            if N_cores <= 25:
    
                for i, (x, y) in enumerate(self.nodes):
    
                    ax.text(
                        x,
                        y,
                        str(i),
                        ha="center",
                        va="center",
                        fontsize=text_size,
                        weight="bold"
                    )
    
            xmin, xmax = self.xlim
            ymin, ymax = self.ylim
            
            dx = xmax - xmin
            dy = ymax - ymin
            
            margin = 0.08
            
            ax.set_xlim(
                xmin - margin * dx,
                xmax + margin * dx
            )
            
            ax.set_ylim(
                ymin - margin * dy,
                ymax + margin * dy
            )
    
            ax.set_aspect("equal")
    
            #
            # Remove all axis clutter
            #
            ax.set_xticks([])
            ax.set_yticks([])
    
            for spine in ax.spines.values():
                spine.set_visible(False)
    
            title = f"M{mode}\nβ={eigvals[mode]:.3f}"
            
            for i, j in degeneracies:
                if mode in (i, j):
                    title += " (deg)"
            
            ax.set_title(
                title,
                fontsize=title_size
            )
    
        #
        # Hide unused panels
        #
        for ax in axes[N_modes:]:
            ax.axis("off")
    
        #
        # Single colorbar
        #
        cbar = fig.colorbar(
            last_scatter,
            ax=axes.tolist(),
            shrink=0.8,
            pad=0.02
        )
    
        cbar.set_label(
            "Phase / Sign"
        )
    
        fig.suptitle(
            f"{N_cores}-Core Supermodes",
            fontsize=14
        )
    
        plt.show()
    
    def browse_modes(self):

        C = self.coupling_matrix()
    
        if len(C) == 0:
            return
    
        eigvals, eigvecs = np.linalg.eigh(C)
    
        idx = np.argsort(eigvals)
    
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
    
        N_modes = len(eigvals)
        N_cores = len(self.nodes)

        edge_pairs = self._edge_pairs(self.geometry_threshold, self.geometry_neighbours)
    
        browser = tk.Toplevel(self.root)
    
        browser.title("Supermode Browser")
        browser.geometry("500x500")
    
        browser.transient(self.root)
    
        current_mode = tk.IntVar(value=0)
    
        #
        # Figure
        #
        fig = Figure(figsize=(3, 3))
    
        ax = fig.add_subplot(111)

        #
        # Top controls
        #
        controls = tk.Frame(browser)
        controls.pack(
            fill="x",
            pady=5
        )
        
        plot_frame = tk.Frame(browser)
        plot_frame.pack(
            fill="both",
            expand=True
        )
        
        canvas = FigureCanvasTkAgg(
            fig,
            master=plot_frame
        )
        
        canvas.get_tk_widget().pack(
            fill="both",
            expand=True
        )
        
        
    

        mode_label = tk.Label(
            controls,
            text="Mode 1/1",
            width=15
        )
        mode_label.pack(side="left", padx=10)
        

    
        #
        # Degeneracy detection
        #
        degeneracies = set()
    
        tol = 1e-6
    
        for i in range(N_modes - 1):
    
            if abs(eigvals[i+1] - eigvals[i]) < tol:
    
                degeneracies.add(i)
                degeneracies.add(i+1)
    
        def goto_mode(mode):
    
            current_mode.set(mode)
        
            draw_mode(mode)
    
        def draw_mode(mode):
    
            ax.clear()
            
            ax.set_position([0.02, 0.02, 0.96, 0.96])
    
            field = eigvecs[:, mode]
    
            amp = np.abs(field)
    
            if amp.max() > 0:
                amp /= amp.max()
    
            field = np.real(field)
    
            if np.max(np.abs(field)) > 0:
                field /= np.max(np.abs(field))
    
            sign = np.zeros_like(field)
    
            sign[field > 1e-8] = -1
            sign[field < -1e-8] = 1
    
            xs = [p[0] for p in self.nodes]
            ys = [p[1] for p in self.nodes]
    
            #
            # Couplings
            #
            for i, j, k in edge_pairs:
    
                x1, y1 = self.nodes[i]
                x2, y2 = self.nodes[j]
    
                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color="lightgray",
                    linewidth=1.5,
                    zorder=0
                )
    
            #
            # Nodes
            #
            scatter = ax.scatter(
                xs,
                ys,
                s=1200 * amp + 100,
                c=sign,
                cmap="bwr",
                vmin=-1,
                vmax=1,
                edgecolors="black",
                linewidths=1
            )
    
            #
            # Labels only for smaller systems
            #
            if N_cores <= 25:
    
                for i, (x, y) in enumerate(self.nodes):
    
                    ax.text(
                        x,
                        y,
                        str(i),
                        ha="center",
                        va="center",
                        fontsize=9,
                        weight="bold"
                    )
    
            #
            # Autoscale from actual geometry
            #
            xmin = min(xs)
            xmax = max(xs)
    
            ymin = min(ys)
            ymax = max(ys)
    
            dx = max(xmax - xmin, 0.1)
            dy = max(ymax - ymin, 0.1)
    
            margin = 0.2
    
            ax.set_xlim(
                xmin - margin * dx,
                xmax + margin * dx
            )
    
            ax.set_ylim(
                ymin - margin * dy,
                ymax + margin * dy
            )
    
            ax.set_aspect(
                "equal",
                adjustable="box"
            )
    
            ax.set_xticks([])
            ax.set_yticks([])
    
            for spine in ax.spines.values():
                spine.set_visible(False)
    
            title = (
                f"Mode {mode+1}/{N_modes}"
                f"\nβ = {eigvals[mode]:.6f}"
            )
    
            if mode in degeneracies:
    
                title += "\n(degenerate)"
    
            ax.set_title(title)
    
            mode_label.config(
                text=f"Mode {mode+1}/{N_modes}"
            )
            
            canvas.draw_idle()

        def previous_mode():
    
            m = current_mode.get()
    
            m = (m - 1) % N_modes
    
            current_mode.set(m)
    
            draw_mode(m)
    
        def next_mode():
    
            m = current_mode.get()
    
            m = (m + 1) % N_modes
    
            current_mode.set(m)
    
            draw_mode(m)
    
        tk.Button(
            controls,
            text="⏮ First",
            command=lambda: goto_mode(0)
        ).pack(side="left", padx=3)
        
        tk.Button(
            controls,
            text="◀ Previous",
            command=previous_mode
        ).pack(side="left", padx=3)
        
        tk.Button(
            controls,
            text="Next ▶",
            command=next_mode
        ).pack(side="left", padx=3)
        
        tk.Button(
            controls,
            text="Last ⏭",
            command=lambda: goto_mode(N_modes - 1)
        ).pack(side="left", padx=3)
        
        
        #
        # Keyboard navigation
        #
        browser.bind(
            "<Left>",
            lambda e: previous_mode()
        )
    
        browser.bind(
            "<Right>",
            lambda e: next_mode()
        )
    
        browser.focus_force()
    
        #
        # Mouse wheel navigation
        #
        browser.bind(
            "<MouseWheel>",
            lambda e:
            next_mode()
            if e.delta < 0
            else previous_mode()
        )
    
    
        browser.after(
            200,
            lambda: draw_mode(0)
        )
             
    def generate_lattice(self):
    
        win = tk.Toplevel(self.root)
    
        win.title("Generate Lattice")
        win.resizable(False, False)
    
        win.transient(self.root)
        win.grab_set()
        win.focus_force()
    
        #
        # Lattice type
        #
        tk.Label(
            win,
            text="Lattice type:"
        ).grid(row=0, column=0, padx=10, pady=5, sticky="w")
    
        lattice_var = tk.StringVar(value="hex")
    
        combo = ttk.Combobox(
            win,
            textvariable=lattice_var,
            values=[
                "hex",
                "square",
                "ring",
                "line"
            ],
            state="readonly",
            width=15
        )
    
        combo.grid(row=0, column=1, padx=10, pady=5)
    
        #
        # Generic coupling
        #
        tk.Label(
            win,
            text="Coupling k:"
        ).grid(row=1, column=0, padx=10, pady=5, sticky="w")
    
        k_var = tk.DoubleVar(value=1.0)
    
        tk.Entry(
            win,
            textvariable=k_var,
            width=10
        ).grid(row=1, column=1, padx=10, pady=5)

        #
        # Pitch (geometry mode only)
        #
        pitch_label = tk.Label(win, text="Core pitch (µm):")
        pitch_var   = tk.StringVar(value="40.0")
        pitch_entry = tk.Entry(win, textvariable=pitch_var, width=10)

        def update_pitch_visibility(*_):
            if self.design_mode == "geometry":
                pitch_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
                pitch_entry.grid(row=2, column=1, padx=10, pady=5)
            else:
                pitch_label.grid_forget()
                pitch_entry.grid_forget()

        update_pitch_visibility()
        self.design_mode_var.trace_add("write", update_pitch_visibility)
    
        #
        # N cores (line/ring)
        #
        n_label = tk.Label(
            win,
            text="Number of cores:"
        )
    
        n_var = tk.IntVar(value=6)
    
        n_entry = tk.Entry(
            win,
            textvariable=n_var,
            width=10
        )
    
        #
        # Rows/Cols (square)
        #
        rows_label = tk.Label(
            win,
            text="Rows:"
        )
    
        rows_var = tk.IntVar(value=3)
    
        rows_entry = tk.Entry(
            win,
            textvariable=rows_var,
            width=10
        )
    
        cols_label = tk.Label(
            win,
            text="Columns:"
        )
    
        cols_var = tk.IntVar(value=3)
    
        cols_entry = tk.Entry(
            win,
            textvariable=cols_var,
            width=10
        )
    
        def update_fields(*args):
    
            lattice = lattice_var.get()
    
            n_label.grid_forget()
            n_entry.grid_forget()
    
            rows_label.grid_forget()
            rows_entry.grid_forget()
    
            cols_label.grid_forget()
            cols_entry.grid_forget()
    
            if lattice in ["line", "ring"]:
    
                n_label.grid(
                    row=3,
                    column=0,
                    padx=10,
                    pady=5,
                    sticky="w"
                )
    
                n_entry.grid(
                    row=3,
                    column=1,
                    padx=10,
                    pady=5
                )
    
            elif lattice == "square":
    
                rows_label.grid(
                    row=3,
                    column=0,
                    padx=10,
                    pady=5,
                    sticky="w"
                )
    
                rows_entry.grid(
                    row=3,
                    column=1,
                    padx=10,
                    pady=5
                )
    
                cols_label.grid(
                    row=4,
                    column=0,
                    padx=10,
                    pady=5,
                    sticky="w"
                )
    
                cols_entry.grid(
                    row=4,
                    column=1,
                    padx=10,
                    pady=5
                )
    
        lattice_var.trace_add(
            "write",
            update_fields
        )
    
        update_fields()
    
        def create():
    
            lattice_type = lattice_var.get()
    
            k = float(k_var.get())
    
            self.save_state()
    
            self.nodes = []
            self.edges = []
            self.core_params = []

            # In geometry mode, use pitch in µm; topology uses normalised units
            if self.design_mode == "geometry":
                try:
                    pitch = float(pitch_var.get())
                except Exception:
                    pitch = 40.0
            else:
                pitch = None   # will use original normalised spacing
    
            #
            # LINE
            #
            if lattice_type == "line":
    
                N = max(2, int(n_var.get()))

                if pitch is not None:
                    spacing = pitch
                    x0 = -(N - 1) * spacing / 2
                else:
                    span    = 1.6
                    spacing = span / max(N - 1, 1)
                    x0      = -span / 2
    
                for n in range(N):
    
                    self.nodes.append(
                        (
                            x0 + n * spacing,
                            0
                        )
                    )
                    self.core_params.append({
                        "radius": self.default_radius,
                        "n_core": self.default_n_core
                    })
    
                for n in range(N - 1):
    
                    self.edges.append(
                        (
                            n,
                            n + 1,
                            k
                        )
                    )
    
            #
            # RING
            #
            elif lattice_type == "ring":
    
                N = max(3, int(n_var.get()))

                if pitch is not None:
                    radius = pitch / (2 * np.sin(np.pi / N))
                else:
                    radius = 0.7
    
                for n in range(N):
    
                    theta = 2 * np.pi * n / N
    
                    self.nodes.append(
                        (
                            radius * np.cos(theta),
                            radius * np.sin(theta)
                        )
                    )
                    self.core_params.append({
                        "radius": self.default_radius,
                        "n_core": self.default_n_core
                    })
    
                for n in range(N):
    
                    self.edges.append(
                        (
                            n,
                            (n + 1) % N,
                            k
                        )
                    )
    
            #
            # SQUARE
            #
            elif lattice_type == "square":
    
                rows = max(1, int(rows_var.get()))
                cols = max(1, int(cols_var.get()))

                if pitch is not None:
                    spacing = pitch
                else:
                    span      = 1.6
                    spacing_x = span / max(cols - 1, 1)
                    spacing_y = span / max(rows - 1, 1)
                    spacing   = min(spacing_x, spacing_y)
    
                x0 = -(cols - 1) * spacing / 2
                y0 = +(rows - 1) * spacing / 2
    
                for r in range(rows):
    
                    for c in range(cols):
    
                        self.nodes.append(
                            (
                                x0 + c * spacing,
                                y0 - r * spacing
                            )
                        )
                        self.core_params.append({
                            "radius": self.default_radius,
                            "n_core": self.default_n_core
                        })
    
                def idx(r, c):
                    return r * cols + c
    
                for r in range(rows):
    
                    for c in range(cols - 1):
    
                        self.edges.append(
                            (
                                idx(r, c),
                                idx(r, c + 1),
                                k
                            )
                        )
    
                for r in range(rows - 1):
    
                    for c in range(cols):
    
                        self.edges.append(
                            (
                                idx(r, c),
                                idx(r + 1, c),
                                k
                            )
                        )
    
            #
            # HEX
            #
            elif lattice_type == "hex":
    
                self.nodes.append((0, 0))
                self.core_params.append({
                    "radius": self.default_radius,
                    "n_core": self.default_n_core
                })

                radius = pitch if pitch is not None else 0.7
    
                for n in range(6):
    
                    theta = n * np.pi / 3
    
                    self.nodes.append(
                        (
                            radius * np.cos(theta),
                            radius * np.sin(theta)
                        )
                    )
                    self.core_params.append({
                        "radius": self.default_radius,
                        "n_core": self.default_n_core
                    })
    
                for n in range(1, 7):
    
                    self.edges.append(
                        (
                            0,
                            n,
                            k
                        )
                    )
    
                for n in range(1, 7):
    
                    self.edges.append(
                        (
                            n,
                            1 + (n % 6),
                            k
                        )
                    )
    
            win.destroy()
    
            self._update_view_limits()
            self.redraw()
    
        tk.Button(
            win,
            text="Generate",
            command=create
        ).grid(
            row=10,
            column=0,
            columnspan=2,
            pady=10
        )
     
    def get_excitation_vector(self, eigvecs):
    
        N = eigvecs.shape[0]
    
        result = {"a0": None}
    
        win = tk.Toplevel(self.root)
    
        win.title("Excitation Builder")
        win.geometry("1200x700")
    
        win.transient(self.root)
        win.grab_set()
    
        excitation_type = tk.StringVar(
            value="core"
        )
    
        #
        # storage
        #
        core_coeffs = np.zeros(N)
    
        if N > 0:
            core_coeffs[0] = 1.0
    
        mode_coeffs = np.zeros(N)
    
        if N > 0:
            mode_coeffs[0] = 1.0
    
        selected_core = tk.IntVar(value=0)
        selected_mode = tk.IntVar(value=0)
    
        #
        # top bar
        #
        top = tk.Frame(win)
        top.pack(fill="x", pady=5)
    
        tk.Label(
            top,
            text="Excitation Type:"
        ).pack(side="left", padx=5)
    
        ttk.Combobox(
            top,
            textvariable=excitation_type,
            values=[
                "core",
                "supermode"
            ],
            state="readonly",
            width=15
        ).pack(side="left")
    
        #
        # main area
        #
        main = tk.Frame(win)
        main.pack(fill="both", expand=True)
    
        left = tk.Frame(main)
        left.pack(
            side="left",
            fill="y",
            padx=5,
            pady=5
        )
    
        center = tk.Frame(main)
        center.pack(
            side="left",
            fill="both",
            expand=True
        )
    
        right = tk.Frame(main)
        right.pack(
            side="left",
            fill="both",
            expand=True
        )
    
        #
        # matplotlib figures
        #
        fig_select = Figure(
            figsize=(4, 4),
            constrained_layout=True
        )
        ax_select = fig_select.add_subplot(111)
    
        canvas_select = FigureCanvasTkAgg(
            fig_select,
            master=center
        )
    
        canvas_select.get_tk_widget().pack(
            fill="both",
            expand=True
        )
    
        fig_preview = Figure(
            figsize=(4, 4),
            constrained_layout=True
        )
        ax_preview = fig_preview.add_subplot(111)
    
        canvas_preview = FigureCanvasTkAgg(
            fig_preview,
            master=right
        )
    
        canvas_preview.get_tk_widget().pack(
            fill="both",
            expand=True
        )
    
        #
        # controls frame
        #
        controls = tk.Frame(left)
        controls.pack(fill="both", expand=True)
    
        amp_var = tk.StringVar(value="1.0")
        sign_var = tk.StringVar(value="+")
    
        coeff_var = tk.StringVar(value="1.0")
    
        cmap = plt.cm.magma
        
    
        #
        # helper
        #
        def current_field():
    
            if excitation_type.get() == "core":
    
                return core_coeffs.astype(complex)
    
            else:
    
                coeffs = mode_coeffs.astype(complex)
    
                return eigvecs @ coeffs
    
        #
        # preview plot
        #
        def draw_preview():
    
            ax_preview.clear()
    
            field = current_field()
    
            amp = np.abs(field)
    
            if amp.max() > 0:
                amp = amp / amp.max()
    
            colors = cmap(amp)
    
            xs = [p[0] for p in self.nodes]
            ys = [p[1] for p in self.nodes]
            
            xmin = min(xs)
            xmax = max(xs)
            
            ymin = min(ys)
            ymax = max(ys)
            
            dx = max(xmax - xmin, 1.0)
            dy = max(ymax - ymin, 1.0)
            
            margin = 0.35
            
            ax_preview.set_xlim(
                xmin - margin * dx,
                xmax + margin * dx
            )
            
            ax_preview.set_ylim(
                ymin - margin * dy,
                ymax + margin * dy
            )
    
            for i, j, k in self._edge_pairs(self.geometry_threshold, self.geometry_neighbours):

                x1, y1 = self.nodes[i]
                x2, y2 = self.nodes[j]

                ax_preview.plot([x1, x2],[y1, y2], color="lightgray", linewidth=1.5, zorder=0)

            ax_preview.scatter(
                    xs,
                    ys,
                    s=800 * amp + 100,
                    c=colors,
                    edgecolors="black"
                )
    
            for i, (x, y) in enumerate(self.nodes):
    
                ax_preview.text(
                    x,
                    y,
                    str(i),
                    ha="center",
                    va="center"
                )
    
            ax_preview.set_title(
                "Launched Field Preview"
            )
    
            ax_preview.set_xticks([])
            ax_preview.set_yticks([])
    
            ax_preview.set_aspect("equal")
    
            canvas_preview.draw_idle()
    
        #
        # draw core editor
        #
        def draw_core_editor():
    
            ax_select.clear()
    
            amps = np.abs(core_coeffs)
    
            if amps.max() > 0:
                amps = amps / amps.max()
    
            xs = [p[0] for p in self.nodes]
            ys = [p[1] for p in self.nodes]
            
            xmin = min(xs)
            xmax = max(xs)
            
            ymin = min(ys)
            ymax = max(ys)
            
            dx = max(xmax - xmin, 1.0)
            dy = max(ymax - ymin, 1.0)
            
            margin = 0.35
            
            ax_select.set_xlim(
                xmin - margin * dx,
                xmax + margin * dx
            )
            
            ax_select.set_ylim(
                ymin - margin * dy,
                ymax + margin * dy
            )
    
            for i, j, k in self._edge_pairs():
    
                x1, y1 = self.nodes[i]
                x2, y2 = self.nodes[j]
    
                ax_select.plot(
                    [x1, x2],
                    [y1, y2],
                    color="lightgray"
                )
    
            field = np.real(core_coeffs)

            amp = np.abs(field)
            
            if amp.max() > 0:
                amp = amp / amp.max()
            
            sign = np.zeros_like(field)
            
            sign[field > 1e-8] = -1
            sign[field < -1e-8] = 1
    
            ax_select.scatter(
                xs,
                ys,
                s=1200 * amp + 100,
                c=sign,
                cmap="bwr",
                vmin=-1,
                vmax=1,
                edgecolors="black",
                linewidths=1
            )
            
            ax_select.scatter(
                xs[selected_core.get()],
                ys[selected_core.get()],
                s=1800,
                facecolors="none",
                edgecolors="gold",
                linewidths=2,
                zorder=10
            )
    
            for i, (x, y) in enumerate(self.nodes):
    
                ax_select.text(
                    x,
                    y,
                    str(i),
                    ha="center",
                    va="center"
                )
    
            ax_select.set_title(
                "Click Core"
            )
    
            ax_select.set_xticks([])
            ax_select.set_yticks([])
    
            ax_select.set_aspect("equal")
    
            canvas_select.draw_idle()
    
        #
        # draw supermode
        #
        def draw_supermode():

            ax_select.clear()
        
            mode = selected_mode.get()
        
            field = eigvecs[:, mode]
        
            #
            # Same normalization as show_supermodes
            #
            amp = np.abs(field)
        
            if amp.max() > 0:
                amp /= amp.max()
        
            field = np.real(field)
        
            if np.max(np.abs(field)) > 0:
                field /= np.max(np.abs(field))
        
            tol = 1e-8
        
            sign = np.zeros_like(field)
        
            sign[field > tol] = -1
            sign[field < -tol] = 1
        
            xs = [p[0] for p in self.nodes]
            ys = [p[1] for p in self.nodes]
        
            #
            # Couplings
            #
            for i, j, k in self.edges:
        
                x1, y1 = self.nodes[i]
                x2, y2 = self.nodes[j]
        
                ax_select.plot(
                    [x1, x2],
                    [y1, y2],
                    color="lightgray",
                    linewidth=1.5,
                    zorder=0
                )
        
            #
            # Nodes
            #
            ax_select.scatter(
                xs,
                ys,
                s=1200 * amp + 100,
                c=sign,
                cmap="bwr",
                vmin=-1,
                vmax=1,
                edgecolors="black",
                linewidths=1
            )
        
            #
            # Labels
            #
            if len(self.nodes) <= 25:
        
                for i, (x, y) in enumerate(self.nodes):
        
                    ax_select.text(
                        x,
                        y,
                        str(i),
                        ha="center",
                        va="center",
                        fontsize=9,
                        weight="bold"
                    )
        
            #
            # Same limits philosophy as show_supermodes
            #
            xmin, xmax = self.xlim
            ymin, ymax = self.ylim
        
            dx = xmax - xmin
            dy = ymax - ymin
        
            margin = 0.08
        
            ax_select.set_xlim(
                xmin - margin * dx,
                xmax + margin * dx
            )
        
            ax_select.set_ylim(
                ymin - margin * dy,
                ymax + margin * dy
            )
        
            ax_select.set_aspect(
                "equal",
                adjustable="box"
            )
        
            ax_select.set_xticks([])
            ax_select.set_yticks([])
        
            for spine in ax_select.spines.values():
                spine.set_visible(False)
        
            ax_select.set_title(
                f"M{mode}"
            )
        
            canvas_select.draw_idle()
    
        #
        # rebuild controls
        #
        def rebuild_controls():
    
            for w in controls.winfo_children():
                w.destroy()
    
            if excitation_type.get() == "core":
    
                tk.Label(
                    controls,
                    text="Selected Core"
                ).pack()
                
                tk.Label(
                    controls,
                    textvariable=selected_core,
                    font=("TkDefaultFont", 11, "bold")
                ).pack()
    
                tk.Label(
                    controls,
                    text="Amplitude"
                ).pack()
    
                tk.Entry(
                    controls,
                    textvariable=amp_var
                ).pack()
    
                tk.Label(
                    controls,
                    text="Sign (+/-)"
                ).pack()
    
                tk.Entry(
                    controls,
                    textvariable=sign_var,
                    width=5
                ).pack()
    
                def update_core():
    
                    idx = selected_core.get()
    
                    try:
                        val = float(
                            amp_var.get()
                        )
                    except:
                        return
    
                    if sign_var.get().strip() == "-":
                        val *= -1
    
                    core_coeffs[idx] = val
    
                    draw_core_editor()
                    draw_preview()
    
                tk.Button(
                    controls,
                    text="Apply",
                    command=update_core
                ).pack(pady=10)
                
    
                draw_core_editor()
    
            else:
    
                tk.Label(
                    controls,
                    text="Supermodes"
                ).pack()
    
                lb = tk.Listbox(
                    controls,
                    height=20
                )
    
                lb.pack(
                    fill="y",
                    expand=True
                )
    
                for i in range(N):
    
                    lb.insert(
                        "end",
                        f"M{i}"
                    )
    
                lb.selection_set(0)
    
                def select_mode(event=None):
    
                    if not lb.curselection():
                        return
    
                    idx = lb.curselection()[0]
    
                    selected_mode.set(idx)
    
                    coeff_var.set(
                        str(mode_coeffs[idx])
                    )
    
                    draw_supermode()
    
                lb.bind(
                    "<<ListboxSelect>>",
                    select_mode
                )
    
                tk.Label(
                    controls,
                    text="Coefficient"
                ).pack()
    
                tk.Entry(
                    controls,
                    textvariable=coeff_var
                ).pack()
    
                def update_mode():
    
                    idx = selected_mode.get()
    
                    try:
                        mode_coeffs[idx] = float(
                            coeff_var.get()
                        )
                    except:
                        return
    
                    draw_preview()
    
                tk.Button(
                    controls,
                    text="Apply"
                ).pack(pady=5)
    
                controls.winfo_children()[-1].configure(
                    command=update_mode
                )
    
                draw_supermode()
    
            draw_preview()
    
        #
        # click cores
        #
        def on_core_click(event):
    
            if excitation_type.get() != "core":
                return
    
            if event.inaxes != ax_select:
                return
    
            x = event.xdata
            y = event.ydata
    
            best = None
            dmin = 1e9
    
            for i, (cx, cy) in enumerate(self.nodes):
    
                d = np.hypot(
                    cx - x,
                    cy - y
                )
    
                if d < dmin:
    
                    dmin = d
                    best = i
    
            if best is None:
                return
    
            selected_core.set(best)
    
            value = core_coeffs[best]
    
            amp_var.set(
                str(abs(value))
            )
    
            sign_var.set(
                "+"
                if value >= 0
                else "-"
            )
            
            draw_core_editor()
    
        canvas_select.mpl_connect(
            "button_press_event",
            on_core_click
        )
    
        excitation_type.trace_add(
            "write",
            lambda *args: rebuild_controls()
        )
    
        
        rebuild_controls()

    
        #
        # launch
        #
        def launch():
    
            result["a0"] = current_field()
    
            win.destroy()
    

        
        tk.Button(
            top,
            text="Launch",
            command=launch
        ).pack()
    
        self.root.wait_window(win)
    
        return result["a0"]       
            
    def propagation_simulator(self):
    
        C = self.coupling_matrix()
    
        if len(C) == 0:
            return
    
        N = len(self.nodes)
    
        #
        # Supermodes
        #
        eigvals, eigvecs = np.linalg.eigh(C)
        #
        # Build excitation
        #
        a0 = self.get_excitation_vector(
            eigvecs
        )
        
        if a0 is None:
            return
        
        launch_label = "Custom Excitation"
    
        #
        # z range
        #
        zmax = simpledialog.askfloat(
            "Propagation Length",
            "Maximum z:",
            initialvalue=10.0,
            parent=self.root
        )
    
        if zmax is None:
            return
    
        nsteps = simpledialog.askinteger(
            "Steps",
            "Number of z samples:",
            initialvalue=1000,
            parent=self.root
        )
    
        if nsteps is None:
            return
    
        #
        # z grid
        #
        zs = np.linspace(
            0,
            zmax,
            nsteps
        )
    
        #
        # Power storage
        #
        powers = np.zeros(
            (N, nsteps)
        )
    
        #
        # Modal coefficients
        #
        coeffs = eigvecs.conj().T @ a0
    
        #
        # Propagation
        #
        for iz, z in enumerate(zs):
    
            phase = np.exp(
                -1j * eigvals * z
            )
    
            a = eigvecs @ (
                coeffs * phase
            )
    
            powers[:, iz] = np.abs(a)**2
    
        #
        # Figure layout
        #
        fig = plt.figure(
            figsize=(12, 5)
        )
    
        gs = fig.add_gridspec(
            1,
            2,
            width_ratios=[3, 1]
        )
    
        ax_power = fig.add_subplot(gs[0])
        ax_fiber = fig.add_subplot(gs[1])
    
        #
        # Colormap
        #
        cmap = plt.cm.magma
    
        colors = cmap(
            np.linspace(0, 1, N)
        )
    
        #
        # Power curves
        #
        for core in range(N):
    
            ax_power.plot(
                zs,
                powers[core],
                color=colors[core],
                linewidth=3,
                alpha=.8,
                label=f"Core {core}"
            )
    
        ax_power.set_xlabel("z")
    
        ax_power.set_ylabel(
            "Power"
        )
    
        ax_power.set_title(
            f"Propagation from {launch_label}"
        )
    
        ax_power.set_xlim(
            0,
            zmax
        )
    
    
        ax_power.grid(
            alpha=0.3
        )
    
    
        #
        # Fiber diagram
        #
        xs = [p[0] for p in self.nodes]
        ys = [p[1] for p in self.nodes]
    
        #
        # Draw couplings
        #
        for i, j, k in self._edge_pairs():
    
            x1, y1 = self.nodes[i]
            x2, y2 = self.nodes[j]
    
            ax_fiber.plot(
                [x1, x2],
                [y1, y2],
                color="lightgray",
                linewidth=2,
                zorder=0
            )
    
        #
        # Draw cores
        #
        ax_fiber.scatter(
            xs,
            ys,
            s=600,
            c=colors,
            edgecolors="black",
            linewidths=1.5,
            zorder=3
        )
    
        #
        # Labels
        #
        for i, (x, y) in enumerate(self.nodes):
    
            ax_fiber.text(
                x,
                y,
                str(i),
                ha="center",
                va="center",
                fontsize=10,
                weight="bold",
                color="white"
            )
    
        #
        # Nice limits
        #
        xmin = min(xs)
        xmax = max(xs)
    
        ymin = min(ys)
        ymax = max(ys)
    
        dx = max(xmax - xmin, 0.1)
        dy = max(ymax - ymin, 0.1)
    
        margin = 0.25
    
        ax_fiber.set_xlim(
            xmin - margin * dx,
            xmax + margin * dx
        )
    
        ax_fiber.set_ylim(
            ymin - margin * dy,
            ymax + margin * dy
        )
    
        ax_fiber.set_aspect(
            "equal"
        )
    
        ax_fiber.set_title(
            "Fiber Layout"
        )
    
        ax_fiber.set_xticks([])
        ax_fiber.set_yticks([])
    
        for spine in ax_fiber.spines.values():
            spine.set_visible(False)
    
        plt.tight_layout()
    
        plt.show()
            
    def animate_propagation(self):
        """Animate power propagation across cores as a function of z."""

        C = self.coupling_matrix()

        if len(C) == 0:
            messagebox.showwarning("No structure", "Add cores and couplings first.")
            return

        N = len(self.nodes)

        # ── Eigenmodes ────────────────────────────────────────────────────────
        eigvals, eigvecs = np.linalg.eigh(C)

        # ── Excitation ────────────────────────────────────────────────────────
        a0 = self.get_excitation_vector(eigvecs)
        if a0 is None:
            return

        # ── z range ───────────────────────────────────────────────────────────
        zmax = simpledialog.askfloat(
            "Propagation Length", "Maximum z:",
            initialvalue=10.0, parent=self.root
        )
        if zmax is None:
            return

        nsteps = simpledialog.askinteger(
            "Steps", "Number of z frames:",
            initialvalue=300, parent=self.root
        )
        if nsteps is None:
            return

        zs = np.linspace(0, zmax, nsteps)

        # ── Compute powers ────────────────────────────────────────────────────
        coeffs = eigvecs.conj().T @ a0
        powers = np.zeros((N, nsteps))   # shape (core, z_frame)

        for iz, z in enumerate(zs):
            phase = np.exp(-1j * eigvals * z)
            a = eigvecs @ (coeffs * phase)
            powers[:, iz] = np.abs(a) ** 2

        # ── Normalise globally (not per-frame) so absolute power is preserved ─
        # Divide by the global max so values stay in [0, 1] across all frames.
        # Then apply a square-root stretch so that low-power cores still show
        # meaningful variation even when total power is large — without hiding
        # the absolute difference between frames.
        global_max = powers.max()
        global_max = max(global_max, 1e-30)
        p_norm = powers / global_max          # [0, 1], preserves relative power
        p_vis  = np.sqrt(p_norm)              # sqrt stretch for scatter visual only

        # ── Fiber layout helpers ───────────────────────────────────────────────
        xs = np.array([p[0] for p in self.nodes])
        ys = np.array([p[1] for p in self.nodes])

        x_range = xs.max() - xs.min() if xs.max() != xs.min() else 0.1
        y_range = ys.max() - ys.min() if ys.max() != ys.min() else 0.1
        margin = 0.12
        xlim_anim = (xs.min() - margin * x_range, xs.max() + margin * x_range)
        ylim_anim = (ys.min() - margin * y_range, ys.max() + margin * y_range)

        fiber_cmap = plt.cm.gist_heat

        # ── Build Tk window ───────────────────────────────────────────────────
        win = tk.Toplevel(self.root)
        win.title("Propagation Animation")

        fig = Figure(figsize=(11, 5))
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 2], wspace=0.05,
                              left=0.07, right=0.97, top=0.92, bottom=0.12)
        ax_power = fig.add_subplot(gs[0])
        ax_fiber = fig.add_subplot(gs[1])

        canvas_anim = FigureCanvasTkAgg(fig, master=win)
        canvas_anim.get_tk_widget().pack(fill="both", expand=True)

        # ── Controls bar ──────────────────────────────────────────────────────
        ctrl = tk.Frame(win)
        ctrl.pack(fill="x", pady=4)

        is_playing = tk.BooleanVar(value=False)
        frame_var = tk.IntVar(value=0)

        btn_play = tk.Button(ctrl, text="▶  Play", width=8)
        btn_play.pack(side="left", padx=6)

        speed_label = tk.Label(ctrl, text="Speed:")
        speed_label.pack(side="left")
        speed_var = tk.DoubleVar(value=30.0)
        speed_scale = tk.Scale(
            ctrl, from_=5, to=200, orient="horizontal",
            variable=speed_var, length=120, label="fps"
        )
        speed_scale.pack(side="left", padx=4)

        slider = tk.Scale(
            ctrl, from_=0, to=nsteps - 1,
            orient="horizontal", variable=frame_var,
            length=380, label="z frame"
        )
        slider.pack(side="left", padx=8, fill="x", expand=True)

        z_label = tk.Label(ctrl, text="z = 0.000", width=14, anchor="w")
        z_label.pack(side="left", padx=6)

        # ── Static power-evolution curves ─────────────────────────────────────
        core_cmap = plt.cm.magma
        core_colors_base = core_cmap(np.linspace(0.15, 0.95, N))
        for core in range(N):
            ax_power.plot(
                zs, powers[core],
                color=core_colors_base[core],
                linewidth=1.5, alpha=0.85,
                label=f"Core {core}"
            )

        ax_power.set_xlabel("z")
        ax_power.set_ylabel("Power")
        ax_power.set_title("Power evolution")
        ax_power.set_xlim(0, zmax)
        ax_power.grid(alpha=0.25)
        if N <= 10:
            ax_power.legend(fontsize=7, loc="upper right")

        # Vertical z-cursor line
        vline = ax_power.axvline(x=0, color="red", linewidth=1.5, linestyle="--")

        # ── Fiber cross-section artists ───────────────────────────────────────
        ax_fiber.set_xlim(*xlim_anim)
        ax_fiber.set_ylim(*ylim_anim)
        ax_fiber.set_aspect("equal", adjustable="box")
        ax_fiber.set_xticks([])
        ax_fiber.set_yticks([])
        for spine in ax_fiber.spines.values():
            spine.set_visible(False)
        ax_fiber.set_title("Fiber cross-section", color="black")
        for item in ([ax_power.title, ax_power.xaxis.label, ax_power.yaxis.label]
                     + ax_power.get_xticklabels() + ax_power.get_yticklabels()):
            item.set_color("black")


        # Draw static coupling lines
        for i, j, _ in self._edge_pairs():
            x1, y1 = self.nodes[i]
            x2, y2 = self.nodes[j]
            ax_fiber.plot([x1, x2], [y1, y2], color="#444444", linewidth=1.5, zorder=0)

        # Size
        s_min = 30
        s_max = 800
        p0_vis = p_vis[:, 0]
        scat = ax_fiber.scatter(
            xs, ys,
            s=s_min + (s_max - s_min) * p0_vis,
            c=p0_vis,
            cmap=fiber_cmap, vmin=0, vmax=1,
            edgecolors="#333333", linewidths=0.8,
            zorder=3
        )

        # Core labels
        if N <= 20:
            for i, (x, y) in enumerate(self.nodes):
                ax_fiber.text(
                    x, y, str(i),
                    ha="center", va="center",
                    fontsize=8, weight="bold", color="white", zorder=4
                )

        # ── Update function ───────────────────────────────────────────────────
        def update_frame(iz):
            iz = int(np.clip(iz, 0, nsteps - 1))
            pv = p_vis[:, iz]

            # Update scatter sizes + colours using sqrt-stretched values
            scat.set_sizes(s_min + (s_max - s_min) * pv)
            scat.set_array(pv)

            # Move z cursor
            vline.set_xdata([zs[iz], zs[iz]])

            z_label.config(text=f"z = {zs[iz]:.3f}")
            canvas_anim.draw_idle()

        # Connect slider
        def on_slider(val):
            if not is_playing.get():
                update_frame(int(val))

        slider.config(command=on_slider)

        # Draw initial frame
        update_frame(0)

        # ── Animation loop ────────────────────────────────────────────────────
        _anim_id = [None]
        _current_frame = [0]

        def step():
            if not is_playing.get():
                return
            _current_frame[0] = (_current_frame[0] + 1) % nsteps
            frame_var.set(_current_frame[0])
            update_frame(_current_frame[0])
            interval = max(5, int(1000 / speed_var.get()))
            _anim_id[0] = win.after(interval, step)

        def toggle_play():
            if is_playing.get():
                # Currently playing → pause
                is_playing.set(False)
                btn_play.config(text="▶  Play")
                if _anim_id[0]:
                    win.after_cancel(_anim_id[0])
            else:
                # Currently paused → play
                is_playing.set(True)
                btn_play.config(text="⏸  Pause")
                _current_frame[0] = frame_var.get()
                step()

        btn_play.config(command=toggle_play)

        def on_close():
            is_playing.set(False)
            if _anim_id[0]:
                win.after_cancel(_anim_id[0])
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)

    def export_figure(self):

        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("PDF", "*.pdf")
            ]
        )
    
        if not filename:
            return
    
        self.fig.savefig(
            filename,
            bbox_inches="tight",
            dpi=300
        )

    def redraw(self):

        self.ax.clear()

        if self.design_mode == "geometry":
            self._redraw_geometry()
        else:
            self._redraw_topology()

        self.ax.set_aspect("equal")
        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)
        self.canvas.draw_idle()
        self.ax.set_title(
            f"Mode: {self.mode}  |  Design: {self.design_mode}"
        )
        self.ax.grid(alpha=.2)

    def _redraw_topology(self):
        """Original topology redraw: edges coloured by coupling strength."""

        if self.edges:
            max_k = max(abs(k) for _, _, k in self.edges)
            norm  = colors.Normalize(vmin=-max_k, vmax=max_k)
            cmap  = cm.get_cmap("managua")
        else:
            max_k = 1

        for i, j, k in self.edges:
            x1, y1 = self.nodes[i]
            x2, y2 = self.nodes[j]
            strength   = abs(k) / max_k
            linewidth  = 2 + 2 * strength
            edge_color = cmap(norm(k))
            self.ax.plot(
                [x1, x2], [y1, y2],
                color=edge_color, linewidth=linewidth, zorder=1
            )
            xm = (x1 + x2) / 2
            ym = (y1 + y2) / 2
            self.ax.text(
                xm, ym, f"{k:.2f}",
                bbox=dict(facecolor="white", alpha=0.3, edgecolor="none"),
                ha="center", va="center", zorder=5
            )

        for i, (x, y) in enumerate(self.nodes):
            self.ax.scatter(
                x, y, s=650,
                color="steelblue", edgecolor="slategray",
                linewidth=1.2, zorder=3
            )
            self.ax.text(
                x, y, str(i),
                ha="center", va="center",
                color="white", weight="bold"
            )

    def _redraw_geometry(self):
        """
        Geometry mode redraw.
        - Canvas units = µm.
        - Cores drawn as circles with actual radius.
        - Edges drawn between all pairs; line thickness encodes coupling.
        - Axis label shows scale in µm.
        """
        import matplotlib.patches as mpatches

        self._ensure_core_params()
        N = len(self.nodes)

        if N == 0:
            self.ax.set_xlabel("x (µm)")
            self.ax.set_ylabel("y (µm)")
            return

        # Compute all pairwise couplings for display
        C = self.coupling_matrix()
        all_k = [C[i, j] for i in range(N) for j in range(i+1, N) if C[i,j] != 0]
        max_k  = max((abs(k) for k in all_k), default=1.0)

        # Draw coupling lines
        k_tol = 0.005
        for i in range(N):
            for j in range(i + 1, N):
                k = C[i, j]
                if k <= k_tol:
                    continue
                x1, y1 = self.nodes[i]
                x2, y2 = self.nodes[j]
                alpha     = 0.5 + 0.5 * (abs(k) / max_k)
                linewidth = 1.0 + 3.0  * (abs(k) / max_k)
                self.ax.plot(
                    [x1, x2], [y1, y2],
                    color="royalblue", linewidth=linewidth,
                    alpha=alpha, zorder=1
                )
                # Label coupling near midpoint
                xm = (x1 + x2) / 2
                ym = (y1 + y2) / 2
                self.ax.text(
                    xm, ym, f"κ={k:.2f}",
                    fontsize=6,
                    alpha=0.25 + 0.65 * (abs(k) / max_k),
                    bbox=dict(facecolor="white", alpha=0.4, edgecolor="none"),
                    ha="center", va="center", zorder=5
                )

        # Draw core circles
        for i, (x, y) in enumerate(self.nodes):
            p    = self.core_params[i]
            r    = p["radius"]   # µm
            nc   = p["n_core"]
            ncl  = self.n_cladding

            # Colour by index contrast
            delta = nc - ncl
            frac  = np.clip(delta / 0.02, 0, 1)   # saturate at Δn=0.02
            face_color = plt.cm.YlOrRd(0.3 + 0.6 * frac)

            circle = mpatches.Circle(
                (x, y), radius=r,
                facecolor=face_color,
                edgecolor="black",
                linewidth=1.2,
                zorder=3
            )
            self.ax.add_patch(circle)

            # MFD halo (faint dashed circle)
            w = self._mode_field_radius(r, nc, ncl)
            halo = mpatches.Circle(
                (x, y), radius=w,
                facecolor="none",
                edgecolor="steelblue",
                linewidth=0.6,
                linestyle="--",
                alpha=0.45,
                zorder=2
            )
            self.ax.add_patch(halo)

            # Core index and label
            self.ax.text(
                x, y, str(i),
                ha="center", va="center",
                fontsize=8, weight="bold", color="white", zorder=4
            )
            # Small info below the core
            lam_um = self.wavelength * 1e6
            NA  = np.sqrt(max(nc**2 - ncl**2, 1e-12))
            V   = (2 * np.pi * r / lam_um) * NA
            self.ax.text(
                x, y - r * 1.55,
                f"a={r:.1f}µm\nn={nc:.4f}\nV={V:.2f}",
                ha="center", va="top",
                fontsize=5.5, color="dimgray", zorder=4
            )

        self.ax.set_xlabel("x (µm)")
        self.ax.set_ylabel("y (µm)")
        
        


if __name__ == "__main__":

    root = tk.Tk()

    app = FiberBuilder(root)

    root.mainloop()