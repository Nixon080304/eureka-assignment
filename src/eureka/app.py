from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

from .controller import Controller
from .ui import MainWindow


def main(port_base=5555):
    app = QApplication([])
    controller = Controller(port_base=port_base)
    window = MainWindow(controller)

    try:
        controller.start()

        window.seed.setValue(18)
        window.show()

        controller.submit("load_scene", scene="01", seed=18)

        return app.exec()

    finally:
        controller.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


if __name__ == "__main__":
    raise SystemExit(main())
