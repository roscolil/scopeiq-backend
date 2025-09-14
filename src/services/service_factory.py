"""Service factory for DynamoDB-based services"""

from src.services.abbreviation import abbreviation_service
from src.services.category import category_service


class ServiceFactory:
    """Factory class to provide DynamoDB-based services"""

    @staticmethod
    def get_abbreviation_service():
        """Get abbreviation service"""
        return abbreviation_service

    @staticmethod
    def get_category_service():
        """Get category service"""
        return category_service


# Convenience functions
def get_abbreviation_service():
    """Get the abbreviation service"""
    return ServiceFactory.get_abbreviation_service()


def get_category_service():
    """Get the category service"""
    return ServiceFactory.get_category_service()
