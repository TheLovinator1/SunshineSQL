from __future__ import annotations

import logging
import sqlite3
import typing
from contextlib import closing
from itertools import count
from typing import Any, Self

counter = count()

if typing.TYPE_CHECKING:
    from collections.abc import Generator


def create_connection() -> sqlite3.Connection:
    """Create SQLite connection.

    Returns:
        sqlite3.Connection: SQLite connection.
    """
    return sqlite3.connect("sunshine.db")


conn: sqlite3.Connection = create_connection()
conn.row_factory = sqlite3.Row

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(message)s")


def sql_run(statement: str, values: Any | None = None) -> int | None:  # noqa: ANN401
    """Execute SQL statement.

    Args:
        statement: SQL statement.
        values: Values to insert into the statement.

    Returns:
        None or last row ID.
    """
    log_msg: str = f"Running SQL: {statement}" + (f" with values: {values}" if values else "")
    logger.debug(log_msg)

    cur: sqlite3.Cursor = conn.cursor()
    cur.execute(statement, values if values is not None else {})
    conn.commit()
    return cur.lastrowid


def sql_select(statement: str, values: Any | None = None) -> Generator[Any, Any, None]:  # noqa: ANN401
    """Execute SQL statement and return results.

    Args:
        statement: SQL statement.
        values: Values to insert into the statement.

    Returns:
        Generator of results.
    """
    log_msg: str = f"Running SQL: {statement}" + (f" with values: {values}" if values else "")
    logger.debug(log_msg)

    with closing(conn.cursor()) as cur:
        cur.execute(statement, values or {})
        yield from cur.fetchall()


class Field:  # noqa: PLW1641
    """Field for Sunshine ORM."""

    # TODO(TheLovinator): Add __hash__ (Fixes PLW1641)  # noqa: TD003

    def __init__(self, name: str, py_type: type) -> None:
        """Initialize Field.

        Args:
            name: Field name.
            py_type: Python type. (str, int, float, etc.)
        """
        self.name: str = name
        self.py_type: type = py_type

    def __repr__(self) -> str:
        """Return string representation.

        For example: <Field(name, str)>
        """
        return f"<Field({self.name}, {self.py_type})>"

    def __set__(self, instance: Self | None, value: Any) -> None:  # noqa: ANN401
        """Set value to instance.

        Args:
            instance: Instance of the class.
            value: Value to set.

        Returns:
            None
        """
        if instance is None:
            return

        # Ensure py_type is a type and a subclass of Sunshine
        if isinstance(self.py_type, type) and issubclass(self.py_type, Sunshine) and isinstance(value, int):
            value = next(self.py_type.select(self.py_type.id == value))
        instance.__dict__["_values"][self.name] = value

    def __get__(self, instance: Self | None, owner: type[Self]) -> Any | Self:  # noqa: ANN401
        """Get value from instance.

        Args:
            instance: Instance of the class.
            owner: Class owner.

        Returns:
            Any: Value of the field.
        """
        if instance:
            return instance.__dict__["_values"].get(self.name)
        return self

    def sql_type(self) -> str:
        """Convert Python type to SQLite type.

        Returns:
            str: SQLite type.
        """
        # We have to return INTEGER if the field is a Sunshine object so we can use it as a foreign key.
        if isinstance(self.py_type, type) and issubclass(self.py_type, Sunshine):
            return "INTEGER"
        return {
            str: "TEXT",
            int: "INTEGER",
            float: "REAL",
        }.get(self.py_type, "BLOB")  # Default to BLOB for unknown types

    def to_sql(self, value: Any) -> Any:  # noqa: ANN401, PLR6301
        """Convert value to SQL.

        If the field is a Sunshine object, return the database ID so we can use it as a foreign key.

        Args:
            value: Value to convert.

        Returns:
            Any: Converted value.
        """
        if isinstance(value, Sunshine):
            return value.id
        return value

    def __eq__(self, value: object) -> Condition:
        """Compare field with value.

        Args:
            value: Value to compare.

        Returns:
            Condition: Comparison condition.
        """
        return Condition("=", self, value)


class Condition:
    """For comparing fields with values."""

    def __init__(self, operator: str, field: Field, value: Any) -> None:  # noqa: ANN401
        """Initialize Condition.

        Args:
            operator: Comparison operator.
            field: Field to compare.
            value: Value to compare.
        """
        self.operator: str = operator
        self.field: Field = field
        self.value = value

    def to_sql(self) -> tuple[str, dict[Any, Any]]:
        """Convert condition to SQL.

        Placeholder needs to be there so we can compare two values with the same field name.

        Returns:
            tuple: SQL statement and values.
        """
        placeholder: str = f"var{next(counter)}"
        return (
            f"{self.field.name} {self.operator} :{placeholder}",
            {placeholder: self.value},
        )

    def __or__(self, other: Condition) -> BoolCondition:
        """Combine conditions with OR.

        Args:
            other: Other condition.

        Returns:
            BoolCondition: Combined conditions.
        """
        return BoolCondition(operator="OR", cond1=self, cond2=other)


class BoolCondition(Condition):
    """For combining conditions with OR or AND."""

    def __init__(self, operator: str, cond1: Condition, cond2: Condition) -> None:
        """Initialize BoolCondition.

        Args:
        operator: AND or OR.
        cond1: First condition.
        cond2: Second condition.
        """
        self.operator: str = operator
        self.cond1: Condition = cond1
        self.cond2: Condition = cond2

    def to_sql(self: BoolCondition) -> tuple[str, dict[Any, Any]]:
        """Convert condition to SQL.

        Returns:
            tuple: SQL statement and values.
        """
        sql1, values1 = self.cond1.to_sql()
        sql2, values2 = self.cond2.to_sql()
        values1.update(values2)
        return f"({sql1} {self.operator} {sql2})", values1


class Sunshine:
    """Sunshine ORM base class."""

    _name: str = ""
    _cols: typing.ClassVar[dict] = {}

    @classmethod
    def __init_subclass__(cls: type[Sunshine]) -> None:
        """Register subclass."""
        cls._name = f"{cls.__name__.lower()}s"
        cols: dict[str, Field] = {name: Field(name, py_type) for name, py_type in cls.__annotations__.items()}
        cls._cols = {**cls._cols, **cols}
        for name, field in cols.items():
            setattr(cls, name, field)

        if "id" not in cls._cols:
            cls.id = Field("id", int)  # type: ignore  # noqa: PGH003

    @classmethod
    def create(cls: type[Sunshine]) -> None:
        """Create SQLite table."""
        logger.debug("Creating table for %s", cls.__name__)
        statement: str = f"CREATE TABLE IF NOT EXISTS {cls._name} (id INTEGER PRIMARY KEY, %s)" % ", ".join(
            f"{name} {field.sql_type()}" for name, field in cls._cols.items()
        )

        sql_run(statement)

    @classmethod
    def delete(cls: type[Sunshine]) -> None:
        """Delete SQLite table."""
        logger.debug("Deleting table for %s", cls.__name__)

        statement: str = f"DROP TABLE IF EXISTS {cls._name}"
        sql_run(statement)

    def __init__(self: Sunshine, **kwargs: Any) -> None:  # noqa: ANN401
        """Initialize instance."""
        self._values: dict[str, None] = {
            "id": None,
        }
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __repr__(self: Sunshine) -> str:
        """Return string representation."""
        values_str = ", ".join(f"{key}={val!r}" for key, val in self._values.items())
        return f"<{self.__class__.__name__}({values_str})>"

    def save(self: Sunshine) -> None:
        """Save instance to database."""
        try:
            values = {name: field.to_sql(getattr(self, name)) for name, field in self._cols.items()}

            if self.id:
                statement: str = f"UPDATE {self._name} SET %s WHERE id=:id" % ", ".join(  # noqa: S608
                    f"{name} = :{name}" for name in self._cols
                )
                sql_run(statement, {"id": self.id, **values})
            else:
                statement: str = f"INSERT INTO {self._name} (%s) VALUES (%s)" % (  # noqa: S608
                    ", ".join(f"{name}" for name in self._cols),
                    ", ".join(f":{name}" for name in self._cols),
                )
                self.id: None | int = sql_run(statement, values)

            logger.debug("Saved %s", self)
        except Exception:
            logger.exception("Error occurred while saving %s", self)
            conn.rollback()
            raise

    @classmethod
    def select(cls: type[Sunshine], where: Condition | Any | None = None) -> Generator[Sunshine, None, None]:  # noqa: ANN401
        """Select all rows from table."""
        logger.debug("Selecting all rows from %s", cls.__name__)
        if where:
            where_sql, values = where.to_sql()
        else:
            where_sql, values = "1=1", {}

        statement: str = f"SELECT * FROM {cls._name} WHERE {where_sql}"  # noqa: S608
        for row in sql_select(statement, values):
            yield cls(**dict(row))


class Bye(Sunshine):
    """Another test class."""

    name: str
    company: str


class Hello(Sunshine):
    """Test class."""

    name: str
    age: int
    color: str
    bye_boi: Bye


if __name__ == "__main__":
    Bye.delete()
    Bye.create()

    bye_me = Bye(
        name="John Doe",
        company="TheLovinator AB",
    )

    bye_me.save()

    # Delete the Hello class to start fresh
    Hello.delete()

    # Migrate the Hello class
    Hello.create()

    hello_me = Hello(
        name="TheLovinator",
        age=1337,
        color="blue",
        bye_boi=bye_me,
    )

    hello_me.save()
    hello_me.name = "TheLovinator2"
    hello_me.save()

    second_me = Hello(
        name="TheLovinator3",
        age=1338,
        color="red",
        bye_boi=bye_me,
    )
    second_me.save()

    swag: typing.Generator[Sunshine, None, None] = Hello.select(where=(second_me.id == hello_me.id))
    logger.debug("Swag: %s", list(swag))

    for num, row in enumerate(
        Hello.select(
            where=(Hello.age == second_me.age) | (Hello.age == hello_me.age),
        ),
    ):
        logger.debug("Row %s: %s", num, row)
