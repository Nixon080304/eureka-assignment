from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

from .runtime import Runtime
from .ui import MainWindow


def main(port_base: int = 5555) -> int:
    app = QApplication([])
    runtime = Runtime(port_base=port_base)
    window = MainWindow(runtime)

    try:
        runtime.start()

        window.seed.setValue(18)
        window.show()

        runtime.submit("load_scene", scene="01", seed=18)

        return app.exec()

    finally:
        runtime.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


if __name__ == "__main__":
    raise SystemExit(main())
