from subprocess import Popen
from pathlib import Path


class TgSender:
    def __init__(self, tg_notify_bot_token, wait_config_timeout=60):
        """
        you must register your telegram ac by pass 5th number code to you bot chat
        process wait this time -> wait_config_timeout, than configure failed
        """
        with open("ts.conf", "w") as file:
            file.write(tg_notify_bot_token)
        try:
            with open("ts.conf", "r") as file:
                p = Popen(['telegram-send', '--configure'], stdin=file)  # something long running
                outs, errs = p.communicate(timeout=wait_config_timeout)
                assert p.wait() == 0
                self.tg_success = True
        except FileNotFoundError:
            print("file not exists")
            self.tg_success = False
            # doesn't exist
        except:
            p.terminate()
            self.tg_success = False

    def send_file_to_tg(self, filename: str, text: str = None) -> None:
        if not self.tg_success:
            return

        try:
            path = Path(filename)
            full_path = path.resolve(strict=True)
            if text is None:
                p = Popen(['telegram-send', '-f', full_path])  # something long running
            else:
                p = Popen(['telegram-send', '-f', full_path, '--caption', text])  # something long running
        except FileNotFoundError:
            print("file not exists")

        except:
            p.terminate()
            # doesn't exist

    def send_message_to_tg(self, text: str = "empty message") -> None:
        if not self.tg_success:
            return

        try:
            p = Popen(['telegram-send', text])  # something long running
        except:
            p.terminate()


if __name__ == '__main__':
    from settings import DotEnvControl

    env_control = DotEnvControl()
    tgs = TgSender(env_control.TG_NOTIFY_BOT_TOKEN)
    tgs.send_message_to_tg()
