import abc
from threading import Thread


class BeanObject:
    def onCreate(self):
        pass

    def created(self):
        # 调用onCreate之后
        pass

    def onDestroy(self):
        pass


class Callable(object):
    def __call__(self, *args, **kwargs):
        pass


class Functor(Callable):
    def __init__(self, listening=None):
        self.listening = listening

    def f(self):
        pass

    @abc.abstractmethod
    def onInterrupted(self):
        pass

    @abc.abstractmethod
    def beforeCall(self):
        pass

    def start(self, delay=None, *args, **kwargs):
        Thread(target=self.__call__, *args, **kwargs)

    def run(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        try:
            self.beforeCall()
            self.f(*args, **kwargs)

        except KeyboardInterrupt as e:
            print(e)
            self.onInterrupted()
        except Exception as e:
            self.onException(e)

    def onException(self, e):
        pass
