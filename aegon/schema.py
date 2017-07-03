class TumblerItem:
    def __init__(self, row):
        self.username = row[0]
        self.id = row[1]
        self.date = row[2]
        self.url = row[3]
        self.content = row[4]
        self.isOwner = row[5]

    def toArray(self):
        return [self.username, self.id, self.date, self.url, self.content, self.isOwner]

class TwitterItem:
    def __init__(self, row):
        self.username = row[0]
        self.content = row[1]
        self.date = row[2]
        self.isOwner = row[3]

    def toArray(self):
        return [self.username, self.content, self.date, self.isOwner]

