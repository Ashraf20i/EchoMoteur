import json
import queue
import threading
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

# On importe le pipeline existant depuis main.py.
# main.py ne lance pas son interface console à l'import grâce au :
# if __name__ == "__main__": main()
from main import REFERENCE_DATABASE, run_pipeline


class EchoMoteurDesktop(tk.Tk):
    """
    Interface desktop simple pour EchoMoteur.

    Objectif :
    - ne pas modifier le noyau d'analyse ;
    - permettre de choisir le véhicule ;
    - choisir un fichier audio ;
    - lancer l'analyse ;
    - afficher verdict, scores et explications ;
    - afficher le chemin du rapport JSON.
    """

    def __init__(self):
        super().__init__()

        self.title("EchoMoteur - Diagnostic acoustique moteur")
        self.geometry("920x680")
        self.minsize(850, 620)

        self.result_queue = queue.Queue()
        self.selected_audio_path = tk.StringVar(value="")
        self.vehicle_id = tk.StringVar(value=self._default_vehicle_id())
        self.status_text = tk.StringVar(value="Prêt.")
        self.report_path = tk.StringVar(value="")

        self._build_ui()
        self._poll_queue()

    # ============================================================
    # Construction UI
    # ============================================================

    def _default_vehicle_id(self) -> str:
        if REFERENCE_DATABASE:
            return list(REFERENCE_DATABASE.keys())[0]
        return ""

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        header = ttk.Frame(self, padding=16)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        title = ttk.Label(
            header,
            text="EchoMoteur",
            font=("Segoe UI", 22, "bold"),
        )
        title.grid(row=0, column=0, sticky="w")

        subtitle = ttk.Label(
            header,
            text="Analyse acoustique moteur par traitement du signal",
            font=("Segoe UI", 11),
        )
        subtitle.grid(row=1, column=0, sticky="w", pady=(4, 0))

        controls = ttk.LabelFrame(self, text="Paramètres d'analyse", padding=16)
        controls.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 12))
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Véhicule :").grid(row=0, column=0, sticky="w", padx=(0, 10))

        vehicle_combo = ttk.Combobox(
            controls,
            textvariable=self.vehicle_id,
            values=list(REFERENCE_DATABASE.keys()),
            state="readonly",
            width=35,
        )
        vehicle_combo.grid(row=0, column=1, sticky="w")

        ttk.Label(controls, text="Audio à analyser :").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=(12, 0))

        audio_entry = ttk.Entry(
            controls,
            textvariable=self.selected_audio_path,
        )
        audio_entry.grid(row=1, column=1, sticky="ew", pady=(12, 0))

        browse_button = ttk.Button(
            controls,
            text="Choisir un fichier",
            command=self.choose_audio_file,
        )
        browse_button.grid(row=1, column=2, padx=(10, 0), pady=(12, 0))

        self.analyze_button = ttk.Button(
            controls,
            text="Lancer l'analyse",
            command=self.start_analysis,
        )
        self.analyze_button.grid(row=2, column=1, sticky="w", pady=(16, 0))

        self.progress = ttk.Progressbar(
            controls,
            mode="indeterminate",
            length=220,
        )
        self.progress.grid(row=2, column=2, sticky="e", pady=(16, 0))

        status_label = ttk.Label(
            controls,
            textvariable=self.status_text,
            foreground="#555555",
        )
        status_label.grid(row=3, column=1, columnspan=2, sticky="w", pady=(10, 0))

        body = ttk.Frame(self, padding=(16, 0, 16, 16))
        body.grid(row=2, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(body)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self.summary_frame = ttk.Frame(self.notebook, padding=16)
        self.details_frame = ttk.Frame(self.notebook, padding=16)
        self.json_frame = ttk.Frame(self.notebook, padding=16)

        self.notebook.add(self.summary_frame, text="Résumé")
        self.notebook.add(self.details_frame, text="Scores")
        self.notebook.add(self.json_frame, text="JSON")

        self._build_summary_tab()
        self._build_details_tab()
        self._build_json_tab()

    def _build_summary_tab(self):
        self.summary_frame.columnconfigure(1, weight=1)

        self.verdict_label = ttk.Label(
            self.summary_frame,
            text="Aucun résultat pour le moment.",
            font=("Segoe UI", 18, "bold"),
        )
        self.verdict_label.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 16))

        labels = [
            ("Score global :", "global_score"),
            ("Ratio médian :", "median_ratio"),
            ("Ratio P90 :", "p90_ratio"),
            ("Pire fenêtre brute :", "worst_window"),
            ("Fenêtres anormales :", "anomalous_windows"),
            ("Confiance :", "confidence"),
            ("Contexte audio :", "context"),
        ]

        self.summary_values = {}

        for index, (label, key) in enumerate(labels, start=1):
            ttk.Label(self.summary_frame, text=label, font=("Segoe UI", 10, "bold")).grid(
                row=index, column=0, sticky="w", pady=4, padx=(0, 12)
            )

            value_label = ttk.Label(self.summary_frame, text="-")
            value_label.grid(row=index, column=1, sticky="w", pady=4)
            self.summary_values[key] = value_label

        ttk.Label(self.summary_frame, text="Explication :", font=("Segoe UI", 10, "bold")).grid(
            row=8, column=0, sticky="nw", pady=(18, 4), padx=(0, 12)
        )

        self.explanation_text = tk.Text(
            self.summary_frame,
            height=10,
            wrap="word",
            borderwidth=1,
            relief="solid",
        )
        self.explanation_text.grid(row=8, column=1, sticky="nsew", pady=(18, 4))
        self.explanation_text.configure(state="disabled")

        self.summary_frame.rowconfigure(8, weight=1)

        ttk.Label(self.summary_frame, text="Rapport JSON :", font=("Segoe UI", 10, "bold")).grid(
            row=9, column=0, sticky="w", pady=(12, 0), padx=(0, 12)
        )

        report_entry = ttk.Entry(self.summary_frame, textvariable=self.report_path)
        report_entry.grid(row=9, column=1, sticky="ew", pady=(12, 0))

    def _build_details_tab(self):
        self.details_frame.columnconfigure(0, weight=1)
        self.details_frame.rowconfigure(0, weight=1)

        columns = ("bloc", "score")
        self.block_tree = ttk.Treeview(
            self.details_frame,
            columns=columns,
            show="headings",
            height=12,
        )

        self.block_tree.heading("bloc", text="Bloc de features")
        self.block_tree.heading("score", text="Score")

        self.block_tree.column("bloc", width=260, anchor="w")
        self.block_tree.column("score", width=120, anchor="center")

        self.block_tree.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(self.details_frame, orient="vertical", command=self.block_tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.block_tree.configure(yscrollcommand=scrollbar.set)

    def _build_json_tab(self):
        self.json_frame.columnconfigure(0, weight=1)
        self.json_frame.rowconfigure(0, weight=1)

        self.json_text = tk.Text(
            self.json_frame,
            wrap="none",
            borderwidth=1,
            relief="solid",
        )
        self.json_text.grid(row=0, column=0, sticky="nsew")

        y_scroll = ttk.Scrollbar(self.json_frame, orient="vertical", command=self.json_text.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        self.json_text.configure(yscrollcommand=y_scroll.set)

        x_scroll = ttk.Scrollbar(self.json_frame, orient="horizontal", command=self.json_text.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.json_text.configure(xscrollcommand=x_scroll.set)

    # ============================================================
    # Actions
    # ============================================================

    def choose_audio_file(self):
        path = filedialog.askopenfilename(
            title="Choisir un fichier audio",
            filetypes=[
                ("Fichiers audio", "*.wav *.mp3 *.flac *.ogg *.m4a *.aac"),
                ("Tous les fichiers", "*.*"),
            ],
        )

        if path:
            self.selected_audio_path.set(path)

    def start_analysis(self):
        vehicle_id = self.vehicle_id.get().strip()
        audio_path = self.selected_audio_path.get().strip()

        if not vehicle_id:
            messagebox.showerror("Erreur", "Aucun véhicule sélectionné.")
            return

        if not audio_path:
            messagebox.showerror("Erreur", "Choisis un fichier audio.")
            return

        if not Path(audio_path).exists():
            messagebox.showerror("Erreur", f"Fichier introuvable :\n{audio_path}")
            return

        self._set_busy(True)
        self.status_text.set("Analyse en cours...")

        thread = threading.Thread(
            target=self._run_analysis_thread,
            args=(vehicle_id, audio_path),
            daemon=True,
        )
        thread.start()

    def _run_analysis_thread(self, vehicle_id: str, audio_path: str):
        try:
            result = run_pipeline(vehicle_id, audio_path)
            self.result_queue.put(("success", result))
        except Exception as exc:
            error = {
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            self.result_queue.put(("error", error))

    def _poll_queue(self):
        try:
            status, payload = self.result_queue.get_nowait()

            self._set_busy(False)

            if status == "success":
                self.status_text.set("Analyse terminée.")
                self.display_result(payload)
            else:
                self.status_text.set("Erreur pendant l'analyse.")
                messagebox.showerror("Erreur d'analyse", payload["message"])
                self._display_json(payload)

        except queue.Empty:
            pass

        self.after(150, self._poll_queue)

    def _set_busy(self, busy: bool):
        if busy:
            self.analyze_button.configure(state="disabled")
            self.progress.start(10)
        else:
            self.analyze_button.configure(state="normal")
            self.progress.stop()

    # ============================================================
    # Affichage résultats
    # ============================================================

    def display_result(self, result: dict):
        if result.get("status") != "success":
            self.verdict_label.configure(text="Analyse impossible")
            self._display_json(result)
            return

        comparison = result.get("comparison", {})
        verdict = result.get("verdict", {})
        context = result.get("context", {})

        self.verdict_label.configure(text=verdict.get("label", "Verdict indisponible"))

        self.summary_values["global_score"].configure(
            text=f"{comparison.get('global_score', 0):.4f}"
        )
        self.summary_values["median_ratio"].configure(
            text=f"{comparison.get('median_ratio', 0):.4f}"
        )
        self.summary_values["p90_ratio"].configure(
            text=f"{comparison.get('p90_ratio', 0):.4f}"
        )
        self.summary_values["worst_window"].configure(
            text=f"{comparison.get('worst_window_ratio_raw', 0):.4f}"
        )

        anomalous = comparison.get("anomalous_windows", 0)
        total = comparison.get("total_windows", 0)
        ratio = comparison.get("anomalous_ratio", 0) * 100
        self.summary_values["anomalous_windows"].configure(
            text=f"{anomalous} / {total} ({ratio:.1f} %)"
        )

        self.summary_values["confidence"].configure(
            text=f"{comparison.get('confidence_score', 0):.1f} %"
        )

        self.summary_values["context"].configure(
            text=f"{context.get('status', '-')} ({context.get('score', 0):.1f} %)"
        )

        explanation_lines = []
        message = verdict.get("message")
        if message:
            explanation_lines.append(message)
            explanation_lines.append("")

        summary = verdict.get("summary")
        if summary:
            explanation_lines.append(summary)
            explanation_lines.append("")

        explanations = verdict.get("explanations", [])
        if explanations:
            explanation_lines.append("Détails :")
            for exp in explanations:
                explanation_lines.append(f"- {exp}")

        self._set_text(self.explanation_text, "\n".join(explanation_lines))

        self._display_block_scores(comparison.get("block_scores", {}))
        self._display_json(result)

        self.report_path.set(result.get("report_path", ""))

    def _display_block_scores(self, block_scores: dict):
        for item in self.block_tree.get_children():
            self.block_tree.delete(item)

        ordered = sorted(block_scores.items(), key=lambda item: item[1], reverse=True)

        for block, score in ordered:
            self.block_tree.insert("", "end", values=(block, f"{score:.4f}"))

    def _display_json(self, data: dict):
        text = json.dumps(data, indent=4, ensure_ascii=False)
        self._set_text(self.json_text, text)

    def _set_text(self, widget: tk.Text, text: str):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")


def main():
    app = EchoMoteurDesktop()
    app.mainloop()


if __name__ == "__main__":
    main()
